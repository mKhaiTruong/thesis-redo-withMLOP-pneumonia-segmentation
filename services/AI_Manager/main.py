import sys
from fastapi import FastAPI
from core.exception import CustomException
from core.prometheus_metrics import instrument_app

from ai_manager.config import ConfigManager
from ai_manager.components.lstm_analyzer import LSTM_Analyzer
from packages.dqn.src.dqn.components.dqn_planner import DQN_Planner
from ai_manager.components.executer import Executer

app = FastAPI()
instrument_app(app, service_name="ai_manager")

"""
    Full MAPE-K loop: Monitor → Analyze → Plan → Execute
    
    M - Monitor: take metrics from Prometheus
    A - Analyze: LSTM predict metrics in the future
    P - Plan:    DQN chooses action
    E - Execute: Action executed
"""

config_manager  = ConfigManager()
lstm_analyzer   = LSTM_Analyzer(config=config_manager.get_lstm_config())
dqn_planner     = DQN_Planner(config=config_manager.get_dqn_config())
executer        = Executer()

@app.post("/run-mape-k")
def run_mape_k():
    try:
        # M - Monitoring
        current_state = lstm_analyzer.monitor()
        
        # A - Analyze
        predicted_state = lstm_analyzer.analyze(current_state)
        
        # P - Plan
        state  = {**current_state, **predicted_state}
        action = dqn_planner.plan(state)
        
        # E - Execute
        result = executer.execute(action)
        
        return {
            "current_state":   current_state,
            "predicted_state": predicted_state,
            "action":          action,
            "result":          result
        }
    except Exception as e:
        raise CustomException(e, sys)
    
@app.post("/train-lstm")
def train_lstm():
    try:
        lstm_analyzer.train()
        return {"status": "LSTM Training completed."}
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/train-dqn")
def train_dqn():
    try:
        dqn_planner.train()
        return {"status": "DQN trained successfully"}
    except Exception as e:
        raise CustomException(e, sys)
    

@app.get("/status")
def status():
    try:
        current_state = lstm_analyzer.monitor()
        return {
            "status":       "ok",
            "system_state": current_state,
            "lstm_ready":   lstm_analyzer.is_ready(),
            "dqn_ready":    dqn_planner.is_ready(),
        }
    except Exception as e:
        raise CustomException(e, sys)