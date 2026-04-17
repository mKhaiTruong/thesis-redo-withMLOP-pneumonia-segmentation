import os, shutil, zipfile

from core.logging import logger
from ingestion.adapters import BaseDataIngestionAdapter

class LocalIngestionAdapter(BaseDataIngestionAdapter):
    def __init__(self, src: str):
        if not os.path.exists(src):
            raise ValueError(f"Source path does not exist: {src}")
        self.src = src
    
    def fetch(self, dst: str) -> None:
        os.makedirs(dst, exist_ok=True)
        
        if self.src.endswith(".zip"):
            logger.info(f"Extracting {self.src} -> {dst}")
            
            with zipfile.ZipFile(self.src, 'r') as z:
                z.extractall(dst)
        else:
            logger.info(f"Copying {self.src} -> {dst}")
            shutil.copytree(self.src, dst, dirs_exist_ok=True)