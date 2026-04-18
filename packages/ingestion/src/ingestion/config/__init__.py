import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from core.utils import read_yaml, create_directories
from core.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH

from ingestion import DataIngestionConfig

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def _parse_data_sources(self):
        raw = os.getenv("DATA_SOURCES", "")
        
        sources = []
        for entry in raw.split(","):
            name, source_type, source = entry.strip().split(":")
            sources.append({
                "name": name,
                "source_type": source_type,
                "source": source
            })
        return sources

    def get_data_ingestion_config(self) -> List[DataIngestionConfig]:
        config = self.config.data_ingestion_config
        create_directories([config.root_dir])
        
        sources = self._parse_data_sources()
        configs = []
        for s in sources:
            configs.append(DataIngestionConfig(
                root_dir        = config.root_dir,
                source_type     = s["source_type"],
                source          = s["source"],
                name            = s["name"],
            ))

        return configs