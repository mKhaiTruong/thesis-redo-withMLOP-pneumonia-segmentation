from core.constants import *
from core.utils import read_yaml, create_directories
from claude_validation import ClaudeParams, ClaudeValidationConfig

class ConfigurationManger:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_claude_config(self) -> ClaudeValidationConfig:
        config = self.config.claude_config
        params = self.params.claude_params
        create_directories([config.root_dir])
        
        return ClaudeValidationConfig(
            root_dir     = Path(config.root_dir),
            history_path = Path(config.root_dir) / "decision_history.json",
            params = ClaudeParams(
                model       = params.model,
                max_tokens  = params.max_tokens,
                confidence_threshold = params.confidence_threshold
            )
        )