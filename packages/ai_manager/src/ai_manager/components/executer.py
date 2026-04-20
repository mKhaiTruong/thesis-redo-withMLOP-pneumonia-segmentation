from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

ORCHESTRATOR_URL = "http://orchestrator:7860"
APP_URL          = "http://app:7860"

ACTIONS = {
    0: "do_nothing",
    1: "trigger_retraining",
    2: "switch_to_lighter_model",
    3: "scale_up_service",
    4: "restart_service",
}

class Executer(BaseAIManagerComponent):
    def run(self, action: str) -> dict:
        logger.info(f"Executing action: {action}")
        
        try:
            if action == "do_nothing":
                return {"status": "no action taken"}
            elif action == "trigger_retraining":
                httpx.post(f"{ORCHESTRATOR_URL}/run/ingestion", timeout=None)
                return {"status": "trigger_retraining"}
            elif action == "switch_to_lighter_model":
                httpx.post(f"{APP_URL}/switch-model", timeout=10)
                return {"status": "switch_to_lighter_model"}
            elif action == "scale_up_service":
                httpx.post(f"{ORCHESTRATOR_URL}/run-pipeline", timeout=None)
                return {"status": "scaling up service: training pipeline triggered"}
            elif action == "restart_service":
                httpx.post(f"{ORCHESTRATOR_URL}/run/data_drift", timeout=None)
                return {"status": "data_drift spotted: service restarted"}
            else:
                return {"status": f"unknown action: {action}"}
        except Exception as e:
            return {"status": f"execute failed: {str(e)}"}