import sys

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.data_ingestion import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        data_ingestion_config = cfg_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.fetch_data()


if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionPipeline()
        data_ingestion.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)