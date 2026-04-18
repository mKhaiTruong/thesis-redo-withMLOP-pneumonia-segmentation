import sys
from fastapi import FastAPI

from core.exception import CustomException
from data_drift.pipeline import DataDriftPipeline

app = FastAPI()
pipeline = DataDriftPipeline()

@app.post("/run-data-drift")
def run_data_drift():
    try: 
        pipeline.main()
        return {"message": "Data-drift started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)