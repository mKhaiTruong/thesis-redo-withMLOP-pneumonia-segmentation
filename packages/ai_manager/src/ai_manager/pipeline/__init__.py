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


CONFIDENCE_THRESHOLD = 0.3
@task(name="plan", retries=2, retry_delay_seconds=5)
def plan(state: dict):
    planner  = ComponentFactory().create("dqn_planner")
    claude   = ComponentFactory().create("claude_validator")
    
    result   = planner.run(state)
    action   = result["action"]
    q_spread = result["q_spread"]
    
    if q_spread < CONFIDENCE_THRESHOLD:
        logger.info(f"DQN not confident (q_spread={q_spread:.3f}), asking Claude...")
        claude_result = claude.run(state=state, action=action, q_spread=q_spread)
        
        if claude_result is None:  
            logger.warning("Claude failed, using DQN action")
            return action
        
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