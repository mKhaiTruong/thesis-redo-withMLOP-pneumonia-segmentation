import torch
import segmentation_models_pytorch as smp

from core.logging import logger
from prepare_base_model import PrepareBaseModelConfig

MODEL_MAP = {
    "unet": smp.Unet,
    "unetpp": smp.UnetPlusPlus,
    "deeplabv3": smp.DeepLabV3,
    "manet": smp.MAnet
}

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.activation = None if self.config.modelArchitecture.activation == "None" else self.config.modelArchitecture.activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _build_model(self):
        model_name = self.config.modelArchitecture.model_name
        if model_name.lower() not in MODEL_MAP:
            raise ValueError(f"Model {model_name} is not supported. Choose from {list(MODEL_MAP.keys())}")
    
        return MODEL_MAP[model_name.lower()](
            encoder_name    = self.config.modelArchitecture.encoder,
            encoder_weights = self.config.modelArchitecture.encoder_weights,
            classes         = self.config.modelArchitecture.classes,
            activation      = self.activation
        ) 

    def main(self):
        base_model = self._build_model().to(self.device) 
        torch.save(base_model, self.config.base_model_path)
        logger.info(f"Base model saved at {self.config.base_model_path}")