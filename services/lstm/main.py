import sys
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app

from lstm.config import ConfigManager
from lstm.components.trainer import LSTM_Trainer
from lstm.components.predictor import LSTM_Predictor_

app = FastAPI()
instrument_app(app, service_name="lstm")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train():
    try:
        config  = ConfigManager()
        trainer = LSTM_Trainer(config=config.get_lstm_config())
        trainer.train()
        return {"status": "LSTM Training completed."}
    except Exception as e:
        raise CustomException(e, sys)

@app.post("/predict")
def predict():
    try:
        config    = ConfigManager()
        predictor = LSTM_Predictor_(
            config=config.get_lstm_config(),
            prometheus_url = "http://prometheus:9090"
        )
        return predictor.predict()
    except Exception as e:
        raise CustomException(e, sys)
    
    
@app.get("/current-state")
def current_state():
    try:
        config    = ConfigManager()
        predictor = LSTM_Predictor_(
            config=config.get_lstm_config(),
            prometheus_url = "http://prometheus:9090"
        )
        return predictor.get_current_state()
    except Exception as e:
        raise CustomException(e, sys)