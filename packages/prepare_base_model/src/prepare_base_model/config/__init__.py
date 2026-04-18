import os
from core.constants import *
from core.utils import read_yaml, create_directories
from prepare_base_model import PrepareBaseModelConfig, ModelArchitecture

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model_config
        params = self.params.prepare_base_model_params
        
        model_dir = os.path.join(config.root_dir, f"{params.model_name}_{params.encoder}")
        create_directories([config.root_dir, model_dir])
        
        modelArchitecture = ModelArchitecture(
            model_architecture = params.model_architecture,
            library = params.library,
            model_name = params.model_name,
            encoder = params.encoder,
            encoder_weights = params.encoder_weights,
            classes = params.classes,
            activation = params.activation,
        )

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(os.path.join(model_dir, "base_model.pth")),
            modelArchitecture=modelArchitecture
        )

        return prepare_base_model_config