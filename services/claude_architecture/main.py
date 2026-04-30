import sys
from pydantic import BaseModel
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from claude_architecture.pipeline import ClaudeArchitecturePipeline

app = FastAPI()
instrument_app(app, service_name="claude_architecture")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-claude-architecture")
def run_claude_architecture():
    try:
        pipeline = ClaudeArchitecturePipeline()
        return pipeline.main()
    except Exception as e:
        raise CustomException(e, sys)