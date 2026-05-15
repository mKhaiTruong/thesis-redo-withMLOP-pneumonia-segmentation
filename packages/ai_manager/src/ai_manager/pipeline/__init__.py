from prefect import flow, task
from core.logging import logger
from ai_manager.components.factory import ComponentFactory
from ai_manager.utils.helpers import _compute_entropy

import time
_last_action_time = {}
_cooldown_seconds = 300

@task(name="monitor", retries=2, retry_delay_seconds=5)
def monitor():
    return ComponentFactory().create("monitor").run()

@task(name="analyze", retries=2, retry_delay_seconds=5)
def analyze():
    return ComponentFactory().create("lstm_analyzer").run()

@task(name="plan")
def plan(state: dict) -> str:
    result  = ComponentFactory().create("dqn_planner").run(state)
    action  = result["action"]
    logger.info(f"DQN action: {action}")
    entropy = _compute_entropy(result["q_values"])
    
    claude_result = ComponentFactory().create("claude_validator").run(
        state=state, action=action, q_spread=entropy
    )
    
    if claude_result:
        logger.info(f"Claude: {claude_result['action']} — {claude_result['reasoning']}")
        return claude_result["action"]
    
    return action

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