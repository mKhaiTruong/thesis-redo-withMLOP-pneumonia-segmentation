import sys
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app

from ai_manager.pipeline import MAPE_K_Pipeline

app = FastAPI()
instrument_app(app, service_name="ai_manager")

"""
    Full MAPE-K loop: Monitor → Analyze → Plan → Execute
    
    M - Monitor: take metrics from Prometheus
    A - Analyze: LSTM predict metrics in the future
    P - Plan:    DQN chooses action
    E - Execute: Action executed
"""

@app.post("/run-mape-k")
def run_mape_k():
    try:
        return MAPE_K_Pipeline().run_once()
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/train-lstm")
def train_lstm():
    try:
        import httpx
        res = httpx.post("http://lstm:7860/train", timeout=None)
        return {"status": "LSTM training triggered", "response": res.status_code}
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/train-dqn")
def train_dqn():
    try:
        import httpx
        res = httpx.post("http://dqn:7860/run-dqn-planner-training", timeout=None)
        return {"status": "DQN training triggered", "response": res.status_code}
    except Exception as e:
        raise CustomException(e, sys)
    

@app.get("/status")
def status():
    try:
        import httpx
        lstm_ready = httpx.get("http://lstm:7860/health", timeout=5).json()["status"] == "ok"
        dqn_ready  = httpx.get("http://dqn:7860/health", timeout=5).json()["status"] == "ok"
        return {
            "status":     "ok",
            "lstm_ready": lstm_ready,
            "dqn_ready":  dqn_ready,
        }
    except Exception as e:
        raise CustomException(e, sys)