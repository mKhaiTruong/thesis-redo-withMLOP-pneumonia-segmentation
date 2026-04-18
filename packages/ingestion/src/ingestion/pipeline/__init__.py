import sys

from core.logging import logger
from core.exception import CustomException
from ingestion.config import ConfigurationManager
from ingestion.components import DataIngestion

class DataIngestionPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        data_ingestion_configs = cfg_manager.get_data_ingestion_config()
        for config in data_ingestion_configs:
            data_ingestion = DataIngestion(config=config)
            data_ingestion.fetch_data()


if __name__ == "__main__":
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)