import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from core.constants import *
from core.utils import read_yaml, create_directories
from transformation import DataTransformationConfig

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation_config
        params = self.params.data_transformation_params
        create_directories([config.root_dir])
        
        sources   = self._parse_data_sources()
        data_dirs = [
            {
                "path": Path(self.config.data_ingestion_config.root_dir) / s["name"],
                "name": s["name"]
            }
            for s in sources
        ]
        
        data_transformation_config = DataTransformationConfig(
            root_dir               = config.root_dir,
            data_dirs              = data_dirs,
            out_train_dir          = config.out_train_dir,
            out_valid_dir          = config.out_valid_dir,
            out_infer_dir          = config.out_infer_dir,
            params_image_size      = params.image_size,
            params_skip_background_ratio = params.skip_background_ratio,
            params_slice_interval  = params.slice_interval,
            params_valid_size      = params.valid_size,
            params_infer_size      = params.infer_size
        )

        return data_transformation_config