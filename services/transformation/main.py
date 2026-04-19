import sys
from fastapi import FastAPI, BackgroundTasks

from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from transformation.pipeline import DataTransformationPipeline

app = FastAPI()
pipeline = DataTransformationPipeline()
instrument_app(app, service_name="transformation")

@app.post("/run-transformation")
def run_transformation():
    try: 
        pipeline.main()
        return {"message": "Transformation started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)