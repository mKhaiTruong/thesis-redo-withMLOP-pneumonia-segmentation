import sys
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from orchestrator.pipeline import OrchestratorPipeline

app = FastAPI()
pipeline = OrchestratorPipeline()
instrument_app(app, service_name="orchestrator")

@app.post("/run-pipeline")
def run_full_pipeline():
    try:
        pipeline.run_full_pipeline()
        return {"status": "pipeline completed"}
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/run/{service_name}")
def run_single_service(service_name: str):
    try:
        pipeline.run_single_service(service_name)
        return {"status": f"{service_name} completed"}
    except Exception as e:
        raise CustomException(e, sys)