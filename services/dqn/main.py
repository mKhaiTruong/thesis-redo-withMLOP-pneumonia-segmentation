import sys
from fastapi import FastAPI

from core.exception import CustomException
from core.prometheus_metrics import instrument_app
from dqn.pipeline import TrainingPipeline

app = FastAPI()
instrument_app(app, service_name="dqn_planner_training")

@app.post("/run-dqn-planner-training")
def run_dqn_planner_training():
    try: 
        pipeline = TrainingPipeline()
        pipeline.main()
        return {"message": "Training DQN Planner started/completed successfully"}
    except Exception as e:
        raise CustomException(e, sys)