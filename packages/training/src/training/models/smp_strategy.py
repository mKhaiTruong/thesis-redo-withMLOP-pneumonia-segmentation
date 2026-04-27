import segmentation_models_pytorch as smp
import torch.nn as nn
from training.models import BuildModelStrategy

SMP_MODELS  = {
    "unet":        smp.Unet,
    "unetpp":      smp.UnetPlusPlus,
    "deeplabv3":   smp.DeepLabV3,
    "segformer":   smp.Segformer,
    "manet":       smp.MAnet,
}

class SMPStrategy(BuildModelStrategy):
    def __init__(self):
        pass
    
    def build_model(self, config) -> nn.Module:
        return SMP_MODELS[config.model.model_name](
            encoder_name    = config.model.encoder,
            encoder_weights = config.model.encoder_weights,
            classes         = 1,
            activation      = None,
        )