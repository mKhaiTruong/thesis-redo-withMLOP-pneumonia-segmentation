import sys

from core.logging import logger
from core.exception import CustomException
from transformation.config import ConfigurationManager
from transformation.components import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        data_transformation_config = cfg_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform()
        data_transformation.push_to_kaggle()


if __name__ == "__main__":
    STAGE_NAME = "Data Transformation Stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation = DataTransformationPipeline()
        data_transformation.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)