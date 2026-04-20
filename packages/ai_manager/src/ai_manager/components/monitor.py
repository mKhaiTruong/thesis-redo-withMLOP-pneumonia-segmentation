from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

LSTM_URL = "http://lstm:7860"

class Monitor(BaseAIManagerComponent):
    def __init__(self, lstm_url: str = LSTM_URL):
        self.url = lstm_url
        
    def run(self) -> dict:
        try:
            return httpx.get(f"{self.url}/current-state", timeout=10).json()
        except Exception:
            logger.warning("Monitor failed — returning zeros")
            return {k: [0.0] for k in [
                "current_cpu", "current_ram", "current_latency", "current_drift"
            ]}