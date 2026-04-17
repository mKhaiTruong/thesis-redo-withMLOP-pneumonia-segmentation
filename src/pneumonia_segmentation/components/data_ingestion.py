import os, sys
from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.entity.entity_config import DataIngestionConfig
from pneumonia_segmentation.adapters.factory import IngestionAdapterFactory

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config  = config
        self.adapter = IngestionAdapterFactory.create_adapter(
            self.config.ingestion_type, self.config.source
        )
        
    def fetch_data(self) -> None:
        try: 
            dst = os.path.join(self.config.root_dir, self.config.name)
            self.adapter.fetch(dst)
            logging.info(f"Data fetched at {dst}")
        except Exception as e:
            raise CustomException(e, sys)