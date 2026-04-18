import httpx
from core.logging import logger

MICROSERVICES = {
    "ingestion":          "http://ingestion:7860/run-ingestion",
    "transformation":     "http://transformation:7860/run-transformation",
    "data_drift":         "http://data_drift:7860/run-data-drift",
    # "prepare_base_model": "http://prepare_base_model:7864/run-prepare-base-model",
}

class OrchestratorPipeline:
    def run_full_pipeline(self):
        for service in MICROSERVICES:
            self.run_single_service(service)

    def run_single_service(self, service: str):
        if service not in MICROSERVICES:
            raise ValueError(f"Unknown service: {service}. Available: {list(MICROSERVICES.keys())}")

        url = MICROSERVICES[service]
        logger.info(f"Triggering {service}...")

        with httpx.Client(timeout=None) as client:
            res = client.post(url)
            res.raise_for_status()
        logger.info(f"{service} completed")
    