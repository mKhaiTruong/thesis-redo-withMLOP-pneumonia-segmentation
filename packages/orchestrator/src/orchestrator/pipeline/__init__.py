import httpx
from core.logging import logger
from orchestrator import MICROSERVICES, APP_URL
from orchestrator.components import retrain_flow, ml_pipeline
from orchestrator.pipeline.action_helpers import _scale_app, _swap_model_version

_current_model = "int8"
_model_history = []
        
class OrchestratorPipeline:
    def __init__(self):
        pass
    
    def run_full_pipeline(self):
        ml_pipeline()

    def run_single_service(self, service: str):
        if service not in MICROSERVICES:
            raise ValueError(f"Unknown service: {service}")
        ml_pipeline(services=[service])
        
    # ACTION BEHAVIOR
    def execute_action(self, action: str) -> dict:
        global _current_model, _model_history
        
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
        
        elif action == "scale_out_service":
            return _scale_app(3)
        elif action == "scale_in_service":
            return _scale_app(1)
        elif action == "swap_model_version":
            target      = "fp32" if _current_model== "int8" else "int8"
            model_file  = "best_model_int8.onnx" if target == "int8" else "best_model.onnx"
            
            _model_history.append(_current_model)
            _current_model = target
            return _swap_model_version(model_file=model_file)
        elif action == "rollback":
            if len(_model_history) > 0:
                prev = _model_history.pop()
                file = "best_model_int8.onnx" if prev == "int8" else "best_model.onnx"
                
                _current_model = prev
                return _swap_model_version(model_file=file)
            return {"status": "NOTHING TO ROLLBACK"}
        else:
            return {"status": f"UNKNOWN ACTION -> {action}"}