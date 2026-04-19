from core.constants import *
from core.utils import read_yaml, create_directories
from lstm import LSTM_Predictor_Config, LSTM_Params_Config

class ConfigManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH, 
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
        
    def get_lstm_config(self) -> LSTM_Predictor_Config:
        config = self.config.lstm_config
        params = self.params.lstm_params
        create_directories([config.root_dir])
        
        return LSTM_Predictor_Config(
            root_dir  = Path(config.root_dir),
            model_dir = Path(config.root_dir) / "lstm_model.pth",
            
            lstm_params = LSTM_Params_Config(
                input_size  = params.input_size,
                hidden_size = params.hidden_size,
                num_layers  = params.num_layers,
                input_steps = params.input_steps,
                output_steps = params.output_steps,
                epochs      = params.epochs,
                batch_size  = params.batch_size,
                lr          = params.lr
            ),
        )