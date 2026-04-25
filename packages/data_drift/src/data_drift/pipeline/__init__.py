import sys, json, numpy as np
from pathlib import Path

from core.logging import logger
from core.exception import CustomException
from core.utils.mlflow_setup import setup_mlflow
from data_drift.config import ConfigurationManager
from data_drift.components import DataDriftDetector

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

setup_mlflow()
class DataDriftPipeline:
    def __init__(self):
        pass
        
    def main(self):
        cfg_manager         = ConfigurationManager()
        data_drift_config   = cfg_manager.get_data_drift_config()
        data_drift          = DataDriftDetector(config=data_drift_config)
        result              = data_drift.run()
        
        if "status" in result:
            logger.info(f"Data drift status: {result['status']}")
            data_drift.push_baseline_to_hf()
            return
        
        report_path = Path("artifacts/data_drift/drift_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)
        
        import mlflow
        with mlflow.start_run(run_name="data_drift"):
            mlflow.log_metric("drift_score", result["drift_score"])
            mlflow.log_metric("is_drift", int(result["is_drift"]))
            mlflow.log_metric("n_images", result["n_images"])
            mlflow.log_param("threshold", result["threshold"])
            mlflow.log_artifact(str(report_path))
        
        data_drift.push_baseline_to_hf()


if __name__ == "__main__":
    STAGE_NAME = "Data drift Stage"
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_drift = DataDriftPipeline()
        data_drift.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
            raise CustomException(e, sys)