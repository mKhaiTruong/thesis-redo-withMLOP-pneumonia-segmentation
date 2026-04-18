import sys
from fastapi import FastAPI

from core.exception import CustomException
from prepare_base_model.pipeline import PrepareBaseModelPipeline

app = FastAPI()
pipeline = PrepareBaseModelPipeline()

@app.post("/run-prepare-base-model")
def run_prepare_base_model():
    try: 
        pipeline.main()
        return {"message": "Prepare Base Model started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)