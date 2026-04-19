import sys
from fastapi import FastAPI

from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from data_drift.pipeline import DataDriftPipeline

app = FastAPI()
pipeline = DataDriftPipeline()
instrument_app(app, service_name="data_drift")

@app.post("/run-data-drift")
def run_data_drift():
    try: 
        pipeline.main()
        return {"message": "Data-drift started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)