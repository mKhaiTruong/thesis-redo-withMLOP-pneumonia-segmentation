from torch.optim.lr_scheduler import OneCycleLR
from training import TrainingConfig

class LR_Scheduler:
    def __init__(self):
        pass
    
    @staticmethod
    def get_lr_scheduler(config: TrainingConfig, loaders, optimizer):
        if config.optimizer.lr_scheduler == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr = config.optimizer.lr,
                steps_per_epoch = len(loaders['train']),
                epochs = config.train.epochs,
                pct_start = 2 / config.train.epochs  # warmup 2 epochs
            )
        else:
            scheduler = None
        
        return scheduler