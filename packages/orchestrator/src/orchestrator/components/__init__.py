import httpx
from prefect import flow, task
from prefect.input import RunInput
from prefect.flow_runs import pause_flow_run
from core.logging import logger
from orchestrator import MICROSERVICES, POLL_TIMEOUT


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
    with httpx.Client(timeout=None) as client:
        client.post(MICROSERVICES[name]).raise_for_status()
        
@flow(name="ml-pipeline", log_prints=True)
def ml_pipeline(services: list[str] | None = None):
    tasks = {
        "ingestion":      run_ingestion,
        "transformation": run_transformation,
        "data_drift":     run_data_drift,
    }
    
    for s in (services or list(tasks.keys())):
        if s in tasks:
            tasks[s]()


class Kaggle_Approval(RunInput):
    confirmed: bool = False
    
@flow(name="retrain-pipeline", log_prints=True)
def retrain_flow(app_url: str):
    # Node 1
    ml_pipeline()
    
    # Node 2
    logger.info("Data ready. Go to Kaggle and train the model.")
    approval = pause_flow_run(
        wait_for_input=Kaggle_Approval,
        timeout=POLL_TIMEOUT
    )
    
    if not approval.confirmed:
        logger.warning("Training not confirmed. Stopping.")
        return {"status": "CANCELLED"}
    
    # Node 3
    httpx.post(f"{app_url}/reload-model", timeout=10)
    return {"status": "RETRAINING COMPLETE"}