import sys

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        data_transformation_config = cfg_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.transform()


if __name__ == "__main__":
    STAGE_NAME = "Data Transformation Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation = DataTransformationPipeline()
        data_transformation.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)