import sys

from core.logging import logger
from core.exception import CustomException
from prepare_base_model.config import ConfigurationManager
from prepare_base_model.components import PrepareBaseModel

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
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        prepare_base_model = PrepareBaseModelPipeline()
        prepare_base_model.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)