import sys, mlflow

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.evaluation import Evaluation

from dotenv import load_dotenv
import os

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_REPO_OWNER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

class EvaluationPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        evaluation_config = cfg_manager.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        
        with mlflow.start_run(run_name="evaluation"):
            metrics = evaluation.validate()
            
            flat_metrics = {}
            for model_name, model_metrics in metrics.items():
                for metric_name, value in model_metrics.items():
                    flat_metrics[f"{model_name}_{metric_name}"] = value

            mlflow.log_metrics(flat_metrics)

if __name__ == "__main__":
    STAGE_NAME = "Evaluation Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        evaluation = EvaluationPipeline()
        evaluation.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
    except Exception as e:
            raise CustomException(e, sys) 