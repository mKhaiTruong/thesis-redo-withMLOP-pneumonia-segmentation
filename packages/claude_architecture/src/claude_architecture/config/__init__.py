from core.constants import *
from core.utils import read_yaml, create_directories
from claude_architecture import LLMParams, ClaudeArchitectureConfig

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_claude_architecture_config(self) -> ClaudeArchitectureConfig:
        config = self.config.claude_config
        params = self.params.claude_params
        create_directories([config.root_dir])
        
        return ClaudeArchitectureConfig(
            root_dir     = Path(config.root_dir),
            history_path = Path(config.root_dir) / "architecture_history.json",
            prometheus_url   = params.prometheus_url,
            orchestrator_url = params.orchestrator_url,
            params = LLMParams(
                model        = params.model,
                max_tokens   = params.max_tokens,
            )
        )