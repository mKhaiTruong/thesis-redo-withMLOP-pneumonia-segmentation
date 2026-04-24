import sys, cv2, io, numpy as np, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from core.exception import CustomException
from inference.pipeline import PredictionPipeline

import psutil
print("========= APP STARTING =========", flush=True)
def check_ram_usage():
    process = psutil.Process()
    process_ram_mb = process.memory_info().rss / 1024 / 1024
    if process_ram_mb > 1800:
        return False, process_ram_mb
    return True, process_ram_mb

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
    try:
        print("Loading model...", flush=True)
        ml_models["model"] = PredictionPipeline()
        print("Model loaded OK", flush=True)
    except Exception as e:
        print(f"STARTUP CRASH: {e}", flush=True)
        import traceback
        traceback.print_exc()
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
from core.prometheus_metrics import instrument_app, DRIFT_SCORE, IS_DRIFT
instrument_app(app, service_name="app")

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML_PAGE

@app.get("/health")
def health():
    return {"status": "ok", "model": "onnx-int8"}

@app.get("/debug")
async def debug():
    import psutil
    process = psutil.Process()
    return {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_mb": psutil.virtual_memory().used / 1024 / 1024,
        "ram_total_mb": psutil.virtual_memory().total / 1024 / 1024,
        "process_ram_mb": process.memory_info().rss / 1024 / 1024,
    }
    
@app.post("/debug/set-drift")
def set_drift(score: float):
    DRIFT_SCORE.set(score)
    IS_DRIFT.set(1 if score > 60 else 0)
    return {"drift_score": score}

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

        DRIFT_SCORE.set(score)
        IS_DRIFT.set(1 if drift_result["is_drift"] else 0)

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
        
        if "System Overloaded" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server temporarily overloaded, please retry in a few seconds"
            )
        raise HTTPException(status_code=500, detail=traceback.format_exc())


# ACTIONS BEHAVIOR
import onnxruntime as ort

@app.post("/reload-model")
def reload_model():
    try:
        ml_models["model"].reload()
        return {"status": "model reloaded"}
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/switch-model")
def switch_model(model_type: str = "int8"):
    try:
        path = f"artifacts/best_model_{model_type}.onnx"
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {path}")
        ml_models["model"].model.session = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        return {"status": "switched", "model": path}
    except Exception as e:
        raise CustomException(e, sys)