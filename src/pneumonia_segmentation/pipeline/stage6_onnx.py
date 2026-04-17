import sys, mlflow

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.onnx import Onnx

class OnnxPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        onnx_config = cfg_manager.get_onnx_config()
        onnx = Onnx(config=onnx_config)
        onnx.export_onnx()
        onnx.quantize()

if __name__ == "__main__":
    STAGE_NAME = "ONNX Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        onnx = OnnxPipeline()
        onnx.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys) 