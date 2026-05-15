import sys, os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from core.logging import logger
from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from lstm.config import ConfigManager
from lstm.components.trainer import LSTM_Trainer
from lstm.components.predictor import LSTM_Predictor_

PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
LSTM_CONFIG     = ConfigManager().get_lstm_config()

predictor: LSTM_Predictor_ | None = None
def _try_load_predictor():
    global predictor
    
    model_path  = LSTM_CONFIG.model_dir
    if model_path.exists():
        try:
            predictor = LSTM_Predictor_(
                config= LSTM_CONFIG,
                prometheus_url = PROMETHEUS_URL
            )
            logger.info("LSTM predictor loaded on startup")
        except Exception as e:
            logger.warning(f"Could not load model on startup: {e}")
            predictor = None
    else:
        logger.info("No model checkpoint found — call /train first")
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    LSTM_CONFIG.root_dir.mkdir(parents=True, exist_ok=True)
    _try_load_predictor()
    yield

app = FastAPI(title="LSTM Predictor Service", lifespan=lifespan)
instrument_app(app, service_name="lstm")

# ___________ENDPOINTS______________
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model_loaded": predictor is not None,
        "prometheus": PROMETHEUS_URL,
    }

@app.post("/train")
def train():
    
    global predictor
    logger.info("Training started")
    
    try:
        trainer = LSTM_Trainer(config=LSTM_CONFIG)
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    try:
        predictor = LSTM_Predictor_(config=LSTM_CONFIG, prometheus_url=PROMETHEUS_URL)
    except Exception as e:
        logger.error(f"Failed to reload predictor after training: {e}")
        raise HTTPException(status_code=500, detail=f"Training done but reload failed: {e}")
        
    logger.info("Training complete — predictor reloaded")
    return {"status": "ok", "message": "Training complete"}

@app.post("/predict")
def predict():
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Call POST /train first."
        )
    
    try:
        result = predictor.predict()
    except Exception as e:
        raise CustomException(e, sys)
    
    return JSONResponse(content=result)
    
    
@app.get("/current-state")
def current_state():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor.get_current_state()