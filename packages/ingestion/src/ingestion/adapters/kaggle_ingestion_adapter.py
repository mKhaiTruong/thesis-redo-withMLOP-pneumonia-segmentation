import os, sys, subprocess
from dotenv import load_dotenv

from core.logging import logger
from core.exception import CustomException
from ingestion.adapters import BaseDataIngestionAdapter

class KaggleIngestionAdapter(BaseDataIngestionAdapter):
    def __init__(self, dataset: str):
        self.dataset = dataset  # format: "owner/dataset-name"
    
    def fetch(self, dst: str) -> None:
        try: 
            load_dotenv()

            os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
            os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN", "")

            import kaggle
            kaggle.api.authenticate()
            
            os.makedirs(dst, exist_ok=True)
            logger.info(f"Downloading Kaggle dataset {self.dataset} -> {dst}")
            
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.dataset, "-p", dst, "--unzip", "--force"],
                check=True
            )
        except Exception as e:
            raise CustomException(e, sys)