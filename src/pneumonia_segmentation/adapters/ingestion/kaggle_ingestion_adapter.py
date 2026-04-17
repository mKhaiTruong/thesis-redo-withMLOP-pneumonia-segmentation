import os, sys, kaggle
import subprocess

from dotenv import load_dotenv
load_dotenv()

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.adapters import BaseDataIngestionAdapter

class KaggleIngestionAdapter(BaseDataIngestionAdapter):
    def __init__(self, dataset: str):
        self.dataset = dataset  # format: "owner/dataset-name"
    
    def fetch(self, dst: str) -> None:
        try: 
            os.makedirs(dst, exist_ok=True)
            logging.info(f"Downloading Kaggle dataset {self.dataset} -> {dst}")
            kaggle.api.authenticate()
            
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", self.dataset, "-p", dst, "--unzip"],
                check=True
            )
        except Exception as e:
            raise CustomException(e, sys)