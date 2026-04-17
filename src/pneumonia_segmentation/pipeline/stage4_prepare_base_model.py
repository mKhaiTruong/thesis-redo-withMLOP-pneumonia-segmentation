import sys

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.prepare_base_model import PrepareBaseModel

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        prepare_base_model_config = cfg_manager.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.main()


if __name__ == "__main__":
    STAGE_NAME = "Prepare Base Model Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelPipeline()
        prepare_base_model.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)