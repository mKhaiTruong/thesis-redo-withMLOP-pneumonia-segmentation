from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

LSTM_URL = "http://lstm:7860"

class Analyzer(BaseAIManagerComponent):
    def __init__(self, lstm_url: str = LSTM_URL):
        self.url = lstm_url
        
    def run(self) -> dict:
        try:
            return httpx.post(f"{self.url}/predict", timeout=30).json()
        except Exception:
            logger.warning("Analyze failed — returning zeros")
            return {k: [0.0]*4 for k in [
                "predicted_cpu", "predicted_ram", "predicted_latency", "predicted_drift"
            ]}