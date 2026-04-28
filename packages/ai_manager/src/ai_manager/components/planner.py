from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

DQN_URL = "http://dqn:7860"

class Planner(BaseAIManagerComponent):
    def __init__(self, dqn_url: str = DQN_URL):
        self.url = dqn_url
        
    def run(self, state: dict) -> dict:
        try:
            return httpx.post(f"{self.url}/plan", json=state, timeout=10).json()
        except Exception:
            logger.warning("Plan failed — do_nothing")
            return {"action": "do_nothing", "q_spread": 1.0}