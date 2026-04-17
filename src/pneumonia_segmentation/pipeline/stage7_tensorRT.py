import sys

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.tensorRT import TensorRT


class TensorRTPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        tensorRT_config = cfg_manager.get_tensorrt_config()
        tensorRT = TensorRT(config=tensorRT_config)
        tensorRT.run()

if __name__ == "__main__":
    STAGE_NAME = "TensorRT Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        tensorRT = TensorRTPipeline()
        tensorRT.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
    except Exception as e:
            raise CustomException(e, sys) 