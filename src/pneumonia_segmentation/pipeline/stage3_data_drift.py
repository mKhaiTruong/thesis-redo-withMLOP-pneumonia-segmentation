import sys, json
from pathlib import Path

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.data_drift_detector import DataDriftDetector

class DataDriftPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager         = ConfigurationManager()
        data_drift_config   = cfg_manager.get_data_drift_config()
        data_drift          = DataDriftDetector(config=data_drift_config)
        result              = data_drift.run()
        
        report_path = Path("artifacts/data_drift/drift_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2)
        
        import mlflow
        with mlflow.start_run(run_name="data_drift"):
            mlflow.log_metric("drift_score", result["drift_score"])
            mlflow.log_metric("is_drift", int(result["is_drift"]))
            mlflow.log_metric("n_images", result["n_images"])
            mlflow.log_param("threshold", result["threshold"])
            mlflow.log_artifact(str(report_path))


if __name__ == "__main__":
    STAGE_NAME = "Data drift Stage"
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_drift = DataDriftPipeline()
        data_drift.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)