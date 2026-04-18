import os, sys, time, torch, json
import torch.nn as nn
import segmentation_models_pytorch as smp

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException

from pneumonia_segmentation.utils.data_loaders import get_dataloaders
from pneumonia_segmentation.utils.helpers.optimizer import Optimizer
from pneumonia_segmentation.utils.helpers.early_stopper import EarlyStopper
from pneumonia_segmentation.utils.helpers.lr_scheduler import LR_Scheduler
from pneumonia_segmentation.utils.engine.engine import train_one_epoch, validate

MODEL_MAP = {
    "unet":        smp.Unet,
    "unetpp":      smp.UnetPlusPlus,
    "deeplabv3":   smp.DeepLabV3,
    "deeplabv3p":  smp.DeepLabV3Plus,
    "manet":       smp.MAnet,
}

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.best_iou = 0.0 if config.metric.metric_mode == "max" else float('inf')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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

    def _get_loss_function(self):
        dice = smp.losses.DiceLoss(mode='binary')
        focal = smp.losses.FocalLoss(mode='binary', alpha=self.config.metric.alpha, gamma=self.config.metric.gamma)
        return lambda preds, masks: dice(preds, masks) + focal(preds, masks)
    
    # ── training loop ─────────────────────────────────────────
    
    def train(self):
        for epoch in range(self.config.train.start_epoch, self.config.train.epochs):
            start = time.time()
            train_loss = train_one_epoch(
                self.model, self.loaders['train'], 
                self.loss_function, self.optimizer, 
                self.scheduler, epoch, self.device
            )
            
            if epoch % 2 == 0:
                valid_loss, mean_iou = self._run_validation(epoch)
                elapsed = time.time() - start
                self._log_epoch(epoch, elapsed, train_loss, valid_loss, mean_iou)

                score = -0.25*train_loss -0.25*valid_loss + 0.5*mean_iou
                self.early_stopper(score)
                if self.early_stopper.early_stop:
                    logging.info("Early stopping triggered.")
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
        logging.info(f"Best model updated — IOU: {self.best_iou:.4f}")
        
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
        logging.info(f"Run info saved at {m.run_info_dir}")
        
    # ── logging ──────────────────────────────────────────────
    
    def _log_epoch(self, epoch: int, elapsed: float, train_loss, valid_loss, mean_iou):
        logging.info(
            f"Epoch {epoch} | {elapsed / 60:.2f} min | "
            f"Train Loss {train_loss:.4f} | "
            f"Valid Loss {valid_loss:.4f} | "
            f"IOU {mean_iou:.4f} | Best {self.best_iou:.4f}"
        )