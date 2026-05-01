from claude_architecture.config import ConfigurationManager
from claude_architecture.components import ClaudeArchitecture

class ClaudeArchitecturePipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        architecture = ClaudeArchitecture(config.get_claude_architecture_config())
        return architecture.get_claude_decision()
    
    def _debug(self):
        config = ConfigurationManager()
        claude_config    = config.get_claude_architecture_config()
        claude_architect = ClaudeArchitecture(config=claude_config)
        claude_architect._get_topology_snapshot = lambda: {
            "services": {
                "app": {"cpu": 20.0, "ram": 400.0, "latency": 0.1, "drift": 90.0},
                "lstm": {"cpu": 10.0, "ram": 200.0},
                "dqn":  {"cpu": 5.0,  "ram": 150.0}
            }
        }
        results = claude_architect.get_claude_decision()
        
        print(results['action'])
        print(results['reasoning'])