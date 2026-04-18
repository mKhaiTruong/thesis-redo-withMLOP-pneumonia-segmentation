import httpx
from prefect import flow, task
from core.logging import logger

import os
os.environ["PREFECT_API_URL"] = "http://prefect:4200/api"

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
def ml_pipeline():
    run_ingestion()
    run_transformation()
    run_data_drift()

class OrchestratorPipeline:
    def run_full_pipeline(self):
        ml_pipeline()

    def run_single_service(self, service: str):
        if service not in MICROSERVICES:
            raise ValueError(f"Unknown service: {service}. Available: {list(MICROSERVICES.keys())}")
        ml_pipeline(services=[service])
    