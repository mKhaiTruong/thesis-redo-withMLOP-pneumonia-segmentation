import sys
from pydantic import BaseModel
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from claude_validation.pipeline import ClaudeValidationPipeline

class ValidationRequest(BaseModel):
    metrics:        dict
    dqn_suggestion: str
    dqn_confidence: float

app = FastAPI()
instrument_app(app, service_name="claude_validation")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/validate")
def debug_validate(request: ValidationRequest):
    try:
        pipeline = ClaudeValidationPipeline(
            metrics        = request.metrics,
            dqn_suggestion = request.dqn_suggestion,
            dqn_confidence = request.dqn_confidence
        )
        return pipeline.main()
    except Exception as e:
        raise CustomException(e, sys)

# bash
# curl -X POST "http://localhost:7871/debug/validate" \
#  -H "Content-Type: application/json" \
#  -d '{"metrics": {"current_cpu": [0.9], "current_ram": [0.85], "current_latency": [0.005], "current_drift": [0.8]}, "dqn_suggestion": "trigger_retraining", "dqn_confidence": 0.3}'

@app.post("/run-claude-validation")
def run_claude_validation(request: ValidationRequest):
    try:
        pipeline = ClaudeValidationPipeline(
            metrics        = request.metrics,
            dqn_suggestion = request.dqn_suggestion,
            dqn_confidence = request.dqn_confidence
        )
        return pipeline.main()
    except Exception as e:
        raise CustomException(e, sys)