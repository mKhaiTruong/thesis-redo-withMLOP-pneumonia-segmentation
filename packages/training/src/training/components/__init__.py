import time, torch, json, mlflow
from dataclasses import replace
from pathlib import Path
import segmentation_models_pytorch as smp

from core.logging import logger
from training import TrainingConfig
from training.utils import get_device
from training.models.registry import get_model
from training.utils.data_loaders import get_dataloaders
from training.utils.helpers.optimizer import Optimizer
from training.utils.helpers.losses import LossRegistry
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.loaders        = get_dataloaders(self.config)
        self.model          = get_model(self.config).to(self.device)
        
        self.optimizer      = Optimizer().get_optim(self.model, self.config)
        self.loss_function  = LossRegistry.build(
            names  = self.config.metric.loss_function,
            config = self.config
        )
        self.early_stopper  = EarlyStopper(
            patience    = self.config.train.patience, 
            mode        = self.config.metric.metric_mode
        )
        self.scheduler      = LR_Scheduler.get_lr_scheduler(
            config      = self.config, 
            loaders     = self.loaders, 
            optimizer   = self.optimizer
        )
        
        # AUTO RESUME
        self._resume_from_checkpoint()
    
    
    # ── resume training from checkpoint ───────────────────────
    def _resume_from_checkpoint(self):
        checkpoint_dir = self.config.model.checkpoint_dir
        if not Path(checkpoint_dir).exists():
            return
        
        checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return
        
        latest        = checkpoints[-1]
        checkpoint    = torch.load(latest, map_location=self.device, weights_only=True)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.best_iou = checkpoint["metric"]
        self._save_best_model(is_better=True)
        
        # Override start_epoch
        self.config = replace(
            self.config, 
            train=replace(self.config.train, start_epoch=checkpoint["epoch"] + 1)
        )
        logger.info(f"Resumed from {latest.name} — epoch {checkpoint['epoch']}")
        logger.info(f"IoU {checkpoint['metric']:.4f}")
        
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
        self._save_run_info()
    
    # ── logging ──────────────────────────────────────────────
    
    def _log_epoch(self, epoch: int, elapsed: float, train_loss, valid_loss, mean_iou):
        logger.info(
            f"Epoch {epoch} | {elapsed / 60:.2f} min | "
            f"Train Loss {train_loss:.4f} | "
            f"Valid Loss {valid_loss:.4f} | "
            f"IOU {mean_iou:.4f} | Best {self.best_iou:.4f}"
        )
        
    def _save_run_info(self):
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
    
    # ── validation ────────────────────────────────────────────
    
    def _run_validation(self, epoch) -> tuple[float, float]:
        valid_loss, mean_iou = validate(self.model, self.loaders['valid'], self.loss_function, epoch, self.device)
        lr = self.scheduler.get_last_lr()[0] if self.config.optimizer.lr_scheduler is not None else self.config.optimizer.lr
        
        self._update_best(mean_iou)
        self._save_checkpoint(epoch, lr, mean_iou)
        return valid_loss, mean_iou
    
    # ── save helpers ─────────────────────────────────────────
    
    def _update_best(self, mean_iou):
        if self.config.metric.metric_mode == "max":
            is_better = mean_iou > self.best_iou
            self.best_iou = max(mean_iou, self.best_iou)
        else:
            is_better = mean_iou < self.best_iou
            self.best_iou = min(mean_iou, self.best_iou)
        self._save_best_model(is_better)
        
    def _save_best_model(self, is_better: bool):
        if is_better:
            torch.save(
                self.model.state_dict(), 
                self.config.model.best_model_dir
            )
            logger.info(f"Best model updated — IOU: {self.best_iou:.4f}")
    
    def _save_checkpoint(self, epoch: int, lr: float, metric: float, keep_last: int = 5):
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