import httpx
from core.logging import logger
from orchestrator import MICROSERVICES, APP_URL
from orchestrator.components import retrain_flow, ml_pipeline

class OrchestratorPipeline:
    def run_full_pipeline(self):
        ml_pipeline()

    def run_single_service(self, service: str):
        if service not in MICROSERVICES:
            raise ValueError(f"Unknown service: {service}")
        ml_pipeline(services=[service])
        
    # ACTION BEHAVIOR
    def execute_action(self, action: str) -> dict:
        if action == "trigger_retraining":
            retrain_flow(app_url=APP_URL)
            return {"status": "RETRAINING TRIGGERED"}
        elif action == "switch_to_lighter_model":
            httpx.post(f"{APP_URL}/switch-model", timeout=10)
            return {"status": "SWITCHING TO LIGHTER MODEL"}
        elif action == "scale_up_service":
            self.run_full_pipeline()
            return {"status": "SCALING UP SERVICE"}
        elif action == "restart_service":
            httpx.post(f"{APP_URL}/reload-model", timeout=10)
            return {"status": "RESTARTING SERVICE"}
        else:
            return {"status": f"UNKNOWN ACTION -> {action}"}