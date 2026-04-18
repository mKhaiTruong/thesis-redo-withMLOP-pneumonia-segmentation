import sys
from fastapi import FastAPI, BackgroundTasks

from core.exception import CustomException
from ingestion.pipeline import DataIngestionPipeline

app = FastAPI()
pipeline = DataIngestionPipeline()

@app.post("/run-ingestion")
def run_ingestion():
    try: 
        pipeline.main()
        return {"message": "Ingestion started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)