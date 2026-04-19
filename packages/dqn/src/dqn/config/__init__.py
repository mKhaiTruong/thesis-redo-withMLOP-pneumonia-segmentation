from core.constants import *
from core.utils import read_yaml, create_directories
from dqn import (
    Simulation_Config,
    DQN_Planner_Config, DuelingDQN_Params_Config, DQN_Params_Config
)

class ConfigManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH, 
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_simulation_config(self) -> Simulation_Config:
        params = self.params.simulation_params
        return Simulation_Config(
            cpu_warning     = params.cpu_warning,
            cpu_critical    = params.cpu_critical,
            ram_warning     = params.ram_warning,
            ram_critical    = params.ram_critical,
            latency_warning = params.latency_warning,
            latency_critical    = params.latency_critical,
            drift_warning   = float(params.drift_warning),
            drift_critical  = float(params.drift_critical)
        )
        
        
    def get_dqn_planner_config(self) -> DQN_Planner_Config:
        config = self.config.dqn_planner_config
        params = self.params.dqn_planner_params
        create_directories([config.root_dir])
        
        return DQN_Planner_Config(
            root_dir  = Path(config.root_dir),
            model_dir = Path(config.root_dir) / "dqn_model.pth",
            
            duel_dqn_params = DuelingDQN_Params_Config(
                state_size  = params.state_size,
                action_size = params.action_size,
                hidden_size = params.hidden_size
            ),
            
            dqn_params = DQN_Params_Config(
                lr         = params.lr,
                gamma      = params.gamma,
                epsilon    = params.epsilon,
                eps_min    = params.epsilon_min,
                eps_decay  = params.epsilon_decay,
                batch_size = params.batch_size,
                target_update_freq = params.target_update_freq
            )
        )