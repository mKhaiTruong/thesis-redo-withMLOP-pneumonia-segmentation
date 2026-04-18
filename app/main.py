import sys, cv2, io, numpy as np, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from prometheus_fastapi_instrumentator import Instrumentator

from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.pipeline.prediction import PredictionPipeline

import psutil

def check_ram_usage():
    vm = psutil.virtual_memory()
    if vm.percent > 85.0:
        return False, vm.percent
    return True, vm.percent

def get_drift_status(score):
    if score < 50:
        return "ok"
    elif score < 60:
        return "warn"
    else:
        return "fail"

from pathlib import Path
from fastapi.responses import HTMLResponse

HTML_PAGE = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = PredictionPipeline()
    yield
    ml_models.clear()

app = FastAPI(
    lifespan=lifespan,
    title="Pneumonia Segmentation API",
    description="UNet++ EfficientNetB3 · COVID-19 CT Scan · ONNX INT8",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus
Instrumentator().instrument(app).expose(app)
from prometheus_client import Gauge, Counter, Histogram
drift_score_gauge = Gauge("model_drift_score", "Euclidean (L2 Norm) vs baseline")
drift_counter     = Counter("model_drift_detected_total", "Total drift detections")
infer_time_hist   = Histogram(
    "model_inference_ms",
    "Inference latency",
    buckets=[5, 10, 20, 50, 100, 200, 500]
)

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx-int8"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        is_safe, ram_val = check_ram_usage()
        if not is_safe:
            raise HTTPException(
                status_code=503, 
                detail=f"System Overloaded: RAM at {ram_val}%"
            )
        
        if not file.content_type or not file.content_type.startswith("image/"):
            return {"error": "File must be an image"}
        
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        t0      = time.perf_counter()
        overlay, drift_result = ml_models["model"].predict(image)
        latency = (time.perf_counter() - t0) * 1000
        
        # Push metrics
        score = drift_result["drift_score"]
        status = get_drift_status(score)

        drift_score_gauge.set(score)
        infer_time_hist.observe(latency)
        if drift_result["is_drift"]:
            drift_counter.inc()

        _, buffer = cv2.imencode(".png", overlay)
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()),
            media_type="image/png",
            headers={
                "X-Inference-Ms":   f"{latency:.1f}",
                "X-Drift-Score" :   f"{score:.4f}",
                "X-Drift-Status":   status,
                "X-Drift-Detected": "1" if drift_result["is_drift"] else "0",
                "Access-Control-Expose-Headers": "X-Inference-Ms, X-Drift-Score, X-Drift-Detected, X-Drift-Status"
            },
        )
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)