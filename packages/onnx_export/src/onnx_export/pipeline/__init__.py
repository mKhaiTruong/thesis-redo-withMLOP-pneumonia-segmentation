import sys

from core.logging import logger
from core.exception import CustomException
from onnx_export.config import ConfigurationManager
from onnx_export.components import Onnx

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
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        onnx = OnnxPipeline()
        onnx.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys) 