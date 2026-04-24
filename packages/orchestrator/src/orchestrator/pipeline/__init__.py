import sys, json, time
import httpx
import asyncio
from prefect import flow, task

from core.logging import logger

import os
os.environ["PREFECT_API_URL"] = "http://prefect:4200/api"


HF_REPO_ID      = "bill123mk/pneumonia-seg-weights"
HF_STATUS_FILE  = "retrain_status.json"
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KERNEL   = os.getenv("KAGGLE_KERNEL_SLUG")
POLL_INTERVAL   = 60 * 2        # Sleep every 2 minutes
POLL_TIMEOUT    = 3600 * 4      # Timeout after 4 hours


MICROSERVICES = {
    "ingestion":          "http://ingestion:7860/run-ingestion",
    "transformation":     "http://transformation:7860/run-transformation",
    "data_drift":         "http://data_drift:7860/run-data-drift",
    # "prepare_base_model": "http://prepare_base_model:7864/run-prepare-base-model",
}
    
@task(name="ingestion", retries=2, retry_delay_seconds=10)
def run_ingestion():
    _call("ingestion")
    
@task(name="transformation", retries=2, retry_delay_seconds=10)
def run_transformation():
    _call("transformation")

@task(name="data_drift", retries=2, retry_delay_seconds=10)
def run_data_drift():
    _call("data_drift")

def _call(name: str):
    url = MICROSERVICES[name]
    logger.info(f"Triggering {name}...")
    with httpx.Client(timeout=None) as client:
        res = client.post(url)
        res.raise_for_status()
    
@flow(name="ml-pipeline", log_prints=True)
def ml_pipeline(services: list[str] | None = None):
    tasks = {
        "ingestion":      run_ingestion,
        "transformation": run_transformation,
        "data_drift":     run_data_drift,
    }
    
    to_run = services if services else list(tasks.keys())
    
    for service in to_run:
        if service in tasks:
            tasks[service]()
        else:
            logger.warning(f"Unknown service: {service}")

APP_URL = "http://app:7860"

class OrchestratorPipeline:
    # DEBUGGING ---------------------------
    def run_full_pipeline(self):
        ml_pipeline()

    def run_single_service(self, service: str):
        if service not in MICROSERVICES:
            raise ValueError(f"Unknown service: {service}. Available: {list(MICROSERVICES.keys())}")
        ml_pipeline(services=[service])
    
    # ACTIONS BEHAVIORS ---------------------------
    def execute_action(self, action: str) -> dict:
        if action == "trigger_retraining":
            """
                Flow: 
                    1. Ingestion (Automatic)
                    2. Transformation (Automatic)
                    3. Training: Dev has to go to the Kaggle notebook to start training
                    4. After training, the model is uploaded to Huggingface
                    5. Pull models and then reload model
            """
            try:
                self.run_single_service("ingestion")
                self.run_single_service("transformation")
            except Exception as e:
                logger.error(f"Retrain data prep failed: {e}")
                return {"status": "UNABLE TO UPLOAD DATA TO HF"}
            
            logger.error(f"DATA UPLOADED; PLEASE GO TO KAGGLE NOTEBOOK AND TRAIN")
            logger.error(f"AFTER TRAINING FINISHED, RELOAD MODEL")
            
            return {"status": "RETRAINING TRIGGERED"}
                
        elif action == "switch_to_lighter_model":
            httpx.post(f"{APP_URL}/switch-model", timeout=10)
            return {"status": "SWITCHING TO LIGHTER MODEL"}
        
        elif action == "scale_up_service":
            self.run_full_pipeline()
            return {"status": "SCALING UP SERVICE"}
        
        elif action == "restart_service":
            httpx.post(f"{APP_URL}/reload-model", timeout=10)
            return {"status": "RESTARTING SERVICE"}
        
        else:
            return {"status": f"UNKNOWN ACTION -> {action}"}
    