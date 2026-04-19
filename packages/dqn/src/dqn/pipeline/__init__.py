from dqn.config import ConfigManager
from dqn.components.dqn_planner import DQNPlanner

class TrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigManager()
        planner = DQNPlanner(cfg_manager.get_dqn_planner_config())
        planner.train(cfg_manager.get_simulation_config())