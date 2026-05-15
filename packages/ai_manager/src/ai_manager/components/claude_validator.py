from ai_manager.components import BaseAIManagerComponent
import httpx, os
from core.logging import logger

CLAUDE_URL = os.getenv("CLAUDE_VALIDATION_URL", "http://claude-validation:7860")

class Claude_Validator(BaseAIManagerComponent):
    def __init__(self, claude_validator_url: str = CLAUDE_URL):
        self.url = claude_validator_url
        
    def run(self, state: dict, action: str, q_spread: float) -> dict:
        try:
            current_metrics = {k: v for k, v in state.items() if k.startswith("current_")}
            return httpx.post(
                f"{CLAUDE_URL}/run-claude-validation",
                json={
                    "metrics":        current_metrics,
                    "dqn_suggestion": action,
                    "dqn_confidence": q_spread
                },
                timeout=30
            ).json()
        except Exception as e:
            logger.warning(f"Claude failed, fallback to DQN: {e}")
            return None