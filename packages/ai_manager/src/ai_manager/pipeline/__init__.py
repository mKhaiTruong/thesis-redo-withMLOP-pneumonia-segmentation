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


def _compute_entropy(q_values: list[float]) -> float:
    import numpy as np
    q = np.array(q_values)
    # Softmax
    q = q - q.max()  # numerical stability
    probs = np.exp(q) / np.exp(q).sum()
    # Entropy normalized 0-1
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    max_entropy = np.log(len(q_values))
    return float(entropy / max_entropy)

CONFIDENCE_THRESHOLD = 0.5
@task(name="plan")
def plan(state: dict) -> str:
    result  = ComponentFactory().create("dqn_planner").run(state)
    action  = result["action"]
    entropy = _compute_entropy(result["q_values"])
    
    claude_result = ComponentFactory().create("claude_validator").run(
        state=state, action=action, q_spread=entropy
    )
    
    if claude_result:
        logger.info(f"Claude: {claude_result['action']} — {claude_result['reasoning']}")
        return claude_result["action"]
    
    return action


import time
_last_action_time = {}
_cooldown_seconds = 300

@task(name="execute", retries=2, retry_delay_seconds=5)
def execute(action: str):
    last = _last_action_time.get(action, 0)
    if (time.time() - last) < _cooldown_seconds:
        logger.info(f"Cooldown active, skipping: {action}")
        return {"status": "COOLDOWN", "action": action}
    
    _last_action_time[action] = time.time()
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