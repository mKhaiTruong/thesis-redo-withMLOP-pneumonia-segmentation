from claude_architecture.config import ConfigurationManager
from claude_architecture.components import ClaudeArchitecture

class ClaudeArchitecturePipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        architecture = ClaudeArchitecture(config.get_claude_architecture_config())
        return architecture.get_claude_decision()