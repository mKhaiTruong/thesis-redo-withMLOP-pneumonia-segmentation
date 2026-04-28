from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

CLAUDE_URL = "http://claude_validation:7860"

class Claude_Validator(BaseAIManagerComponent):
    def __init__(self, claude_validator_url: str = CLAUDE_URL):
        self.url = claude_validator_url
        
    def run(self, state: dict, action: str, q_spread: float) -> dict:
        try:
            return httpx.post(
                f"{CLAUDE_URL}/run-claude-validation",
                json={
                    "metrics":        state,
                    "dqn_suggestion": action,
                    "dqn_confidence": q_spread
                },
                timeout=30
            ).json()
        except Exception as e:
            logger.warning(f"Claude failed, fallback to DQN: {e}")
            return None