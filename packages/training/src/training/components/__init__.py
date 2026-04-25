import time, torch, json, mlflow
from pathlib import Path
import torch.nn as nn
import segmentation_models_pytorch as smp

from core.logging import logger
from training import TrainingConfig
from training.utils import get_device
from training.utils.data_loaders import get_dataloaders
from training.utils.helpers.optimizer import Optimizer
from training.utils.helpers.early_stopper import EarlyStopper
from training.utils.helpers.lr_scheduler import LR_Scheduler
from training.utils.engine.engine import train_one_epoch, validate

MODEL_MAP = {
    "unet":        smp.Unet,
    "unetpp":      smp.UnetPlusPlus,
    "deeplabv3":   smp.DeepLabV3,
    "segformer":   smp.Segformer,
    "manet":       smp.MAnet,
    "sam2unet":    None,
}

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_iou = 0.0 if config.metric.metric_mode == "max" else float('inf')
        self.device = get_device()
        
        self.loaders        = get_dataloaders(self.config)
        self.model          = self._get_model()
        self.optimizer      = Optimizer().get_optim(self.model, self.config)
        self.loss_function  = self._get_loss_function()
        self.early_stopper  = EarlyStopper(
            patience    = self.config.train.patience, 
            mode        = self.config.metric.metric_mode
        )
        self.scheduler      = LR_Scheduler.get_lr_scheduler(
            config      = self.config, 
            loaders     = self.loaders, 
            optimizer   = self.optimizer
        )
    
    def _get_model(self) -> nn.Module:
        model_name = self.config.model.model_name.lower()
        
        if model_name == "sam2unet":
            return self._get_foundation_model()
        
        if model_name not in MODEL_MAP:
            raise ValueError(
                f"Model '{model_name}' not supported. Choose from {list(MODEL_MAP.keys())}"
            )
        model = MODEL_MAP[model_name](
            encoder_name    = self.config.model.encoder,
            encoder_weights = self.config.model.encoder_weights,
            classes         = 1,
            activation      = None,
        )
        return model.to(self.device)

    def _get_foundation_model(self) -> nn.Module:
        from sam2unet.SAM2UNet import SAM2UNet
        self.sam2_path = Path("sam2_hiera_large.pt")
        
        logger.info("Loading foundation model: SAM2-UNet")
        logger.info(f"Checkpoint: {str(self.sam2_path)}")
        
        # Freeze backbone, train adapters only
        model = SAM2UNet(model_cfg="sam2_hiera_l.yaml", checkpoint_path=self.sam2_path)
        for name, param in model.named_parameters():
            if "prompt_learn" not in name and "side" not in name and "head" not in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        
        return model.to(self.device)

    def _get_loss_function(self):
        dice = smp.losses.DiceLoss(mode='binary')
        focal = smp.losses.FocalLoss(mode='binary', alpha=self.config.metric.alpha, gamma=self.config.metric.gamma)
        return lambda preds, masks: dice(preds, masks) + focal(preds, masks)
    
    # ── training loop ─────────────────────────────────────────
    
    def train(self):
        mlflow.log_params({
            "lr"        : self.config.optimizer.lr,
            "batch_size": self.config.train.batch_size,
            "epochs"    : self.config.train.epochs,
            "loss_function": self.config.metric.loss_function,
        })
        mlflow.log_param("model_name", self.config.model.model_name)
        mlflow.log_param("encoder", self.config.model.encoder)
        
        self.start_epoch = self.config.train.start_epoch
        for epoch in range(self.start_epoch, self.config.train.epochs):
            start = time.time()
            train_loss = train_one_epoch(
                self.model, self.loaders['train'], 
                self.loss_function, self.optimizer, 
                self.scheduler, epoch, self.device
            )
            
            if epoch % 2 == 0 and epoch > 0:
                valid_loss, mean_iou = self._run_validation(epoch)
                
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "iou": mean_iou,
                }, step=epoch)
                elapsed = time.time() - start
                self._log_epoch(epoch, elapsed, train_loss, valid_loss, mean_iou)

                score = -0.25*train_loss -0.25*valid_loss + 0.5*mean_iou
                self.early_stopper(score)
                if self.early_stopper.early_stop:
                    logger.info("Early stopping triggered.")
                    break

        torch.save(self.model.state_dict(), self.config.model.latest_model_dir)
        self.save_run_info()
    
    # ── validation ────────────────────────────────────────────
    
    def _run_validation(self, epoch) -> tuple[float, float]:
        valid_loss, mean_iou = validate(self.model, self.loaders['valid'], self.loss_function, epoch, self.device)
        lr = self.scheduler.get_last_lr()[0] if self.config.optimizer.lr_scheduler is not None else self.config.optimizer.lr
        
        self._update_best(mean_iou)
        self.save_checkpoint(epoch, lr, mean_iou)
        return valid_loss, mean_iou
    
    def _update_best(self, mean_iou):
        if self.config.metric.metric_mode == "max":
            is_better = mean_iou > self.best_iou
            self.best_iou = max(mean_iou, self.best_iou)
        else:
            is_better = mean_iou < self.best_iou
            self.best_iou = min(mean_iou, self.best_iou)
        self.save_best_model(is_better)
    
    # ── save helpers ─────────────────────────────────────────
    
    def save_checkpoint(self, epoch: int, lr: float, metric: float, keep_last: int = 5):
        ckpt_dir = self.config.model.checkpoint_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr":                   lr,
                "metric":               metric,
            },
            ckpt_dir / f"checkpoint_epoch_{epoch}.pth",
        )

        for old_ckpt in sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))[:-keep_last]:
            old_ckpt.unlink()
    
    def save_best_model(self, is_better: bool):
        if is_better:
            torch.save(
                self.model.state_dict(), 
                self.config.model.best_model_dir
            )
            logger.info(f"Best model updated — IOU: {self.best_iou:.4f}")
        
    def save_run_info(self):
        m = self.config.model
        m.run_info_dir.parent.mkdir(parents=True, exist_ok=True)
        
        run_info = {
            "model_name": m.model_name,
            "encoder":    m.encoder,
            "iou_score":  self.best_iou,
            "status":     "success",
        }
        
        with open(m.run_info_dir, "w") as f:
            json.dump(run_info, f, indent=4)
        logger.info(f"Run info saved at {m.run_info_dir}")
        
    # ── logging ──────────────────────────────────────────────
    
    def _log_epoch(self, epoch: int, elapsed: float, train_loss, valid_loss, mean_iou):
        logger.info(
            f"Epoch {epoch} | {elapsed / 60:.2f} min | "
            f"Train Loss {train_loss:.4f} | "
            f"Valid Loss {valid_loss:.4f} | "
            f"IOU {mean_iou:.4f} | Best {self.best_iou:.4f}"
        )
        
# Sanity check
if __name__=="__main__":
    from pneumonia_segmentation.config import ConfigurationManager
    
    cfg_manager = ConfigurationManager()
    training = Training(cfg_manager.get_training_config())
    loader = training.loaders['train']
    batch = next(iter(loader))
    images, masks = batch['image'], batch['mask']

    print(images.shape)   # expect (B, 3, H, W)
    print(masks.shape)    # expect (B, 1, H, W) 
    print(masks.unique()) # expect tensor([0., 1.])
    
    training.train()