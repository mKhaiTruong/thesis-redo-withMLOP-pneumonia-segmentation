import sys, torch
from fastapi import FastAPI

from core.exception import CustomException
from onnx_export.pipeline import OnnxPipeline

app = FastAPI()
pipeline = OnnxPipeline()

@app.post("/run-onnx_export")
def run_onnx_export():
    try: 
        pipeline.main()
        return {"message": "ONNX started/completed successfully"}
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