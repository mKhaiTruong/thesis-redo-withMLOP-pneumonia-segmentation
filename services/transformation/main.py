import sys
from fastapi import FastAPI, BackgroundTasks

from core.exception import CustomException
from transformation.pipeline import DataTransformationPipeline

app = FastAPI()
pipeline = DataTransformationPipeline()

@app.post("/run-transformation")
def run_transformation():
    try: 
        pipeline.main()
        return {"message": "Transformation started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)