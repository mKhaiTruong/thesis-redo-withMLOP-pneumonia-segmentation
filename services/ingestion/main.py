import sys
from fastapi import FastAPI, BackgroundTasks

from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from ingestion.pipeline import DataIngestionPipeline

app = FastAPI()
pipeline = DataIngestionPipeline()
instrument_app(app, service_name="ingestion")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-ingestion")
def run_ingestion():
    try: 
        pipeline.main()
        return {"message": "Ingestion started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)