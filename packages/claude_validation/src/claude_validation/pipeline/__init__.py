from claude_validation.config import ConfigurationManger
from claude_validation.components import ClaudeValidation

class ClaudeValidationPipeline:
    def __init__(self, metrics: dict, dqn_suggestion: str, dqn_confidence: float):
        self.metrics     = metrics
        self.suggestions = dqn_suggestion
        self.conf        = dqn_confidence
        
    def main(self):
        config          = ConfigurationManger()
        claude_validate = ClaudeValidation(config.get_claude_config())
        
        return claude_validate.claude_validate(
            self.metrics, self.suggestions, self.conf
        )