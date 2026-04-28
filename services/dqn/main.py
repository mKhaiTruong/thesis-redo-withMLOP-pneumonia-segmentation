import sys
from fastapi import FastAPI

from core.exception import CustomException
from core.prometheus_metrics import instrument_app

app = FastAPI()
instrument_app(app, service_name="dqn_planner_training")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run-dqn-planner-training")
def run_dqn_planner_training():
    try:
        from dqn.pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.main()
        return {"message": "Training DQN Planner started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/plan")
def plan(state: dict):
    try:
        from dqn.pipeline import PlanningPipeline
        pipeline = PlanningPipeline(state=state)
        return pipeline.main()
    except Exception as e:
        raise CustomException(e, sys)