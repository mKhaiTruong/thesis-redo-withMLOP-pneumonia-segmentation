import sys, mlflow

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.training import Training

from dotenv import load_dotenv
import os 

load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_REPO_OWNER"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

class TrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager = ConfigurationManager()
        training_config = cfg_manager.get_training_config()
        training = Training(config=training_config)
        
        with mlflow.start_run(run_name="training"):
            training.train()


if __name__ == "__main__":
    STAGE_NAME = "Training Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        training = TrainingPipeline()
        training.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)