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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-mape-k")
def run_mape_k():
    try:
        return MAPE_K_Pipeline().run_once()
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/run-claude-architecture")
def run_claude_architecture():
    try:
        import httpx
        res = httpx.post("http://claude-architecture:7860/run-claude-architecture", timeout=60)
        return res.json()
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
        lstm_ready  = httpx.get("http://lstm:7860/health", timeout=5).json()["status"] == "ok"
        dqn_ready   = httpx.get("http://dqn:7860/health", timeout=5).json()["status"] == "ok"
        claude_ready = httpx.get(
            "http://claude-architecture:7860/health", timeout=5).json()["status"] == "ok"
        
        return {
            "status":       "ok",
            "lstm_ready":   lstm_ready,
            "dqn_ready":    dqn_ready,
            "claude_ready": claude_ready
        }
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/debug/set-drift")
def set_drift(score: float):
    import httpx
    res = httpx.post(f"http://app:7860/debug/set-drift", params={"score": score}, timeout=10)
    return res.json()

@app.get("/metrics/current")
def get_current_metrics():
    try:
        import httpx
        res = httpx.get("http://lstm:7860/current-state", timeout=5)
        return res.json()
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/debug/trigger-scenario")
def trigger_scenario(scenario: str):
    """
    scenario: "spawn" | "swap" | "rollback"
    """
    
    try: 
        import httpx, subprocess
        
        LOCUST_HOST = "http://pneumonia.local"
        LOCUST_FILE = "tests/locustfile_wsl.py"

        scenario_config = {
            "spawn":    {"drift": 10,  "users": 150, "rate": 10},
            "swap":     {"drift": 75,  "users": 30,  "rate": 2},
            "rollback": {"drift": 45,  "users": 60,  "rate": 4},
        }
        
        cfg = scenario_config.get(scenario)
        if not cfg:
            return {"error": f"Unknown scenario: {scenario}"}
        
        # 1. Set drift
        httpx.post(
            f"http://app:7860/debug/set-drift",
            params  = {"score": cfg["drift"]},
            timeout = 5
        )
        
        # 2. Trigger claude_architecture
        result = httpx.post(
            "http://claude-architecture:7860/run-claude-architecture",
            timeout=60
        ).json()
        
        return {
            "scenario": scenario,
            "config":   cfg,
            "claude_decision": result
        }
    except Exception as e:
        raise CustomException(e, sys)
    