from torch.optim import AdamW
from training import TrainingConfig

class Optimizer:
    def __init__(self):
        pass
    
    def get_optim(self, model, config: TrainingConfig):
        try:
            params = [
                {'params': model.decoder.parameters(),          'lr': config.optimizer.lr},
                {'params': model.encoder.parameters(),          'lr': config.optimizer.lr},
                {'params': model.segmentation_head.parameters(),'lr': config.optimizer.lr},
            ]
            
            return AdamW(params, lr=config.optimizer.lr, weight_decay=config.optimizer.decay)
        except Exception:
            return AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.decay)