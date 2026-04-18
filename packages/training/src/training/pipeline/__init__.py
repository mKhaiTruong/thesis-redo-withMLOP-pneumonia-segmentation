import sys

from core.logging import logger
from core.exception import CustomException
from core.utils.mlflow_setup import setup_mlflow
from training.config import ConfigurationManager
from training.components import Training

setup_mlflow()
class TrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        import mlflow
        cfg_manager = ConfigurationManager()
        training_config = cfg_manager.get_training_config()
        training = Training(config=training_config)
        
        with mlflow.start_run(run_name="training"):
            training.train()


if __name__ == "__main__":
    STAGE_NAME = "Training Stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        training = TrainingPipeline()
        training.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)