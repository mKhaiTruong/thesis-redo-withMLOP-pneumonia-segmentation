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