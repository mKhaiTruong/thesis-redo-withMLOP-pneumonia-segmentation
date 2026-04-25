import sys, json, torch
import segmentation_models_pytorch as smp
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

from core.logging import logger
from core.exception import CustomException
from onnx_export import OnnxConfig

MODEL_MAP = {
    "unet":        smp.Unet,
    "unetpp":      smp.UnetPlusPlus,
    "deeplabv3":   smp.DeepLabV3,
    "segformer":   smp.Segformer,
    "manet":       smp.MAnet,
    "sam2unet":    None,
}

class Onnx:
    def __init__(self, config: OnnxConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model  = self._load_model()
        
    def _load_model(self) -> torch.nn.Module:
        try:
            run_info = json.loads(
                Path(self.config.trained_model.run_info_dir).read_text()
            )
            model_name = run_info["model_name"].lower()
            if model_name not in MODEL_MAP:
                raise ValueError(f"Model '{model_name}' not in MODEL_MAP")
            
            # if model_name == "sam2unet":
            #     return self._load_foundation_model()
            
            model = MODEL_MAP[model_name](
                encoder_name    = run_info["encoder"],
                encoder_weights = None,
                classes         = 1,
                activation      = None,
            )
            model.load_state_dict(torch.load(
                self.config.trained_model.best_model_dir,
                map_location = self.device,
                weights_only = True,
            ))
            logger.info(f"Loaded {model_name} + {run_info['encoder']} from best_model.pth")
            return model.eval().to(self.device)
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # def _load_foundation_model(self):
    #     try:
    #         from sam2unet.SAM2UNet import SAM2UNet
        
    #         model = SAM2UNet(model_cfg="sam2_hiera_l.yaml")  # init architecture
    #         model.load_state_dict(torch.load(
    #             self.config.trained_model.best_model_dir,
    #             map_location=self.device,
    #             weights_only=True,
    #         ))
            
    #         logger.info("Loaded SAM2-UNet trained weights")
    #         return model.eval().to(self.device)
    #     except Exception as e:
    #         raise CustomException(e, sys)
    
    def export_onnx(self):
        try: 
            self.config.onnx_model_dir.parent.mkdir(parents=True, exist_ok=True)
            dummy_input = torch.randn(
                1, 3, self.config.image_size, self.config.image_size
            ).to(self.device)
            
            export_model = self.model
            if isinstance(self.model, SAM2UNet):
                class _Wrapper(torch.nn.Module):
                    def forward(self, x):
                        return self.model(x)[0]
                export_model        = _Wrapper()
                export_model.model  = self.model
            
            torch.onnx.export(
                export_model, dummy_input,
                str(self.config.onnx_model_dir),
                opset_version = 17,
                dynamo=False,
                input_names   = ["input"],
                output_names  = ["output"],
                dynamic_axes  = {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            logger.info(f"ONNX exported -> {self.config.onnx_model_dir}")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def quantize(self):
        try: 
            quantize_dynamic(
                model_input = str(self.config.onnx_model_dir),
                model_output= str(self.config.onnx_int8_model_dir),
                weight_type = QuantType.QUInt8
            )
            logger.info(f"INT8 model -> {self.config.onnx_int8_model_dir}")
            
        except Exception as e:
            raise CustomException(e, sys)