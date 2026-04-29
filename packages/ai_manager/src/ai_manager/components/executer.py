from ai_manager.components import BaseAIManagerComponent
import httpx
from core.logging import logger

ORCHESTRATOR_URL = "http://orchestrator:7860"
APP_URL          = "http://app:7860"

ACTIONS = {
    0: "do_nothing",
    1: "trigger_retraining",
    2: "switch_to_lighter_model",
    3: "scale_up_service",      # legacy
    4: "restart_service",
    5: "scale_out_service",     # new
    6: "scale_in_service",      # new
    7: "swap_model_version",    # new
}

class Executer(BaseAIManagerComponent):
    def run(self, action: str) -> dict:
        logger.info(f"Executing action: {action}")
        if action not in ACTIONS.values():
            raise ValueError(f"Unknown action: {action}. Available: {list(ACTIONS.values())}")
        
        try:
            if action == "do_nothing":
                return {"status": "no action taken"}
            
            res = httpx.post(
                f"{ORCHESTRATOR_URL}/execute/{action}", 
                timeout=None
            )
            return res.json()
        except Exception as e:
            return {"status": f"execute failed: {str(e)}"}