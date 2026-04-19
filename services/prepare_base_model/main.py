import sys
from fastapi import FastAPI

from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from prepare_base_model.pipeline import PrepareBaseModelPipeline

app = FastAPI()
pipeline = PrepareBaseModelPipeline()
instrument_app(app, service_name="prepare-base-model")

@app.post("/run-prepare-base-model")
def run_prepare_base_model():
    try: 
        pipeline.main()
        return {"message": "Prepare Base Model started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)