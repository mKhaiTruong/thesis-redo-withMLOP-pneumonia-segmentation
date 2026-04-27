import segmentation_models_pytorch as smp
import torch.nn as nn

class LossRegistry:
    _registry = {}
    
    @classmethod
    def registry(cls, name: str):
        def decorator(loss_cls):
            cls._registry[name] = loss_cls
            return loss_cls
        
        return decorator

    @classmethod
    def build(cls, names: list[str], config) -> nn.Module:
        losses = []
        for name in names:
            if name not in cls._registry:
                raise ValueError(f"Loss '{name}' not found. Available: {list(cls._registry.keys())}")
            losses.append(cls._registry[name](config))
        
        if len(losses) == 1:
            return losses[0]
        return CombinedLoss(losses)

class CombinedLoss(nn.Module):
    def __init__(self, losses: list[nn.Module]):
        super().__init__()
        self.losses = nn.ModuleList(losses)

    def forward(self, preds, masks):
        return sum(loss(preds, masks) for loss in self.losses) / len(self.losses)


@LossRegistry.registry("focal-loss")
class FocalLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = smp.losses.FocalLoss(
            mode='binary',
            alpha=config.metric.alpha,
            gamma=config.metric.gamma
        )

    def forward(self, preds, masks):
        return self.loss(preds, masks)

@LossRegistry.registry("dice-loss")
class DiceLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode='binary')

    def forward(self, preds, masks):
        return self.loss(preds, masks)

@LossRegistry.registry("bce-loss")
class BCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, masks):
        return self.loss(preds, masks)