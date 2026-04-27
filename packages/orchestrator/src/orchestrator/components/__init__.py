import httpx
from prefect import flow, task
from prefect.input import RunInput
from prefect.flow_runs import pause_flow_run
from core.logging import logger
from orchestrator import MICROSERVICES, APP_URL, POLL_TIMEOUT


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
def retrain_flow(app_url: str = APP_URL):
    # Node 1
    ml_pipeline()
    
    # Node 2
    import os
    from prefect import get_run_logger
    
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("✅ DATA READY — Ingestion + Transformation + Drift done")
    logger.info("📋 Next steps:")
    logger.info("   1. Go to Kaggle notebook and train:")
    logger.info(f"   🔗 https://www.kaggle.com/code/{os.getenv('KAGGLE_USERNAME')}/{os.getenv('KAGGLE_KERNEL_SLUG')}")
    logger.info("   2. After training completes, come back here and Resume")
    logger.info("=" * 50)
    
    approval = pause_flow_run(
        wait_for_input=Kaggle_Approval,
        timeout=POLL_TIMEOUT
    )
    
    if not approval.confirmed:
        logger.info("❌ Training not confirmed. Stopping.")
        return {"status": "CANCELLED"}
    logger.info("✅ Training confirmed — reloading model...")
    
    # Node 3
    try:
        httpx.post(f"{app_url}/reload-model", timeout=10)
        logger.info("✅ Model reloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not reload model: {e} — reload manually via /reload-model")