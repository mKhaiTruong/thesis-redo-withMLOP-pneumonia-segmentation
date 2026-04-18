import sys, torch
from fastapi import FastAPI

from core.exception import CustomException
from training.pipeline import TrainingPipeline

app = FastAPI()
pipeline = TrainingPipeline()

@app.post("/run-training")
def run_training():
    try: 
        pipeline.main()
        return {"message": "Training started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)
    
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }