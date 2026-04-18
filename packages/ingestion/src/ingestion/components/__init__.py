import os, sys
from core.logging import logger
from core.exception import CustomException
from ingestion import DataIngestionConfig
from ingestion.adapters.factories import IngestionAdapterFactory

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config  = config
        self.adapter = IngestionAdapterFactory.create_adapter(
            self.config.source_type, self.config.source
        )
        
    def fetch_data(self) -> None:
        try: 
            dst = os.path.join(self.config.root_dir, self.config.name)
            self.adapter.fetch(dst)
            logger.info(f"Data fetched at {dst}")
        except Exception as e:
            raise CustomException(e, sys)