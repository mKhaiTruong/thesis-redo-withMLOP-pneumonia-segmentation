from prefect import flow, task
from core.logging import logger
from ai_manager.components.factory import ComponentFactory

import os
os.environ["PREFECT_API_URL"] = "http://prefect:4200/api"

@task(name="monitor", retries=2, retry_delay_seconds=5)
def monitor():
    return ComponentFactory().create("monitor").run()

@task(name="analyze", retries=2, retry_delay_seconds=5)
def analyze():
    return ComponentFactory().create("lstm_analyzer").run()

@task(name="plan", retries=2, retry_delay_seconds=5)
def plan(state: dict):
    return ComponentFactory().create("dqn_planner").run(state)

@task(name="execute", retries=2, retry_delay_seconds=5)
def execute(action: str):
    return ComponentFactory().create("executer").run(action)


@flow(name="mape-k", log_prints=True)
def mape_k_flow():
    current_state   = monitor()
    predicted_state = analyze()
    action          = plan({**current_state, **predicted_state})
    result          = execute(action)
    
    logger.info(f"MAPE-K completed: action={action}, result={result}")
    return {
        "current_state":    current_state,
        "predicted_state":  predicted_state,
        "action":           action,
        "result":           result  
    }

class MAPE_K_Pipeline:
    def run_once(self) -> dict:
        return mape_k_flow()
    
    def run_loop(self) -> dict:
        from datetime import timedelta
        return mape_k_flow.serve(
            name="mape-k-loop",
            interval=timedelta(minutes=5)
        )