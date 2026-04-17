import os, sys
from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.entity.entity_config import DataIngestionConfig
from pneumonia_segmentation.adapters import BaseDataIngestionAdapter
from pneumonia_segmentation.adapters.local_ingestion_adapter import LocalIngestionAdapter
from pneumonia_segmentation.adapters.kaggle_ingestion_adapter import KaggleIngestionAdapter

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config  = config
        self.adapter = self._get_ingestion_adapter()
    
    def _get_ingestion_adapter(self) -> BaseDataIngestionAdapter:
        source_type = self.config.source_type
        source      = self.config.source
        
        if source_type == "LOCAL":
            return LocalIngestionAdapter(source)
        elif source_type == "KAGGLE":
            return KaggleIngestionAdapter(source)
        else:
            raise ValueError(f"Source type {source_type} not supported")
        
    def fetch_data(self) -> None:
        try: 
            dst = os.path.join(self.config.root_dir, self.config.name)
            self.adapter.fetch(dst)
            logging.info(f"Data fetched at {dst}")
        except Exception as e:
            raise CustomException(e, sys)