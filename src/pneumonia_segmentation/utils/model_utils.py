import torch
import segmentation_models_pytorch as smp

MODEL_MAP = {
    "unetpp": smp.UnetPlusPlus,
    "unet": smp.Unet,
    "deeplabv3plus": smp.DeepLabV3Plus,
}

def build_model(params_model_name, params_encoder, params_encoder_weights,
                    params_classes, params_activation):
    
    if params_model_name not in MODEL_MAP:
        raise ValueError(f"Model {params_model_name} is not supported. Choose from {list(MODEL_MAP.keys())}")
    
    return MODEL_MAP[params_model_name](
        encoder_name    = params_encoder,
        encoder_weights = params_encoder_weights,
        classes         = params_classes,
        activation      = params_activation
    )