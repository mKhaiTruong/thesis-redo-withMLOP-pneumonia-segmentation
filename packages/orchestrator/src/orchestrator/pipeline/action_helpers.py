import os, httpx, docker
from core.logging import logger
from orchestrator import APP_URL

# Scale app
def _scale_app(replicas: int) -> dict: 
    try:
        client = docker.from_env()
        
        existing = client.containers.list(filters={"name": "pneumonia-segmentation-app"})
        if not existing:
            return {"status": "SCALE FAILED — no app container found"}
        
        template = existing[0]
        current_count = len(existing)
        
        if replicas > current_count:
            stopped = client.containers.list(
                all=True, 
                filters={"name": "pneumonia-segmentation-app", "status": "exited"}
            )
            for c in stopped:
                c.remove()
                
            for i in range(current_count+1, replicas+1):
        
                client.containers.run(
                    image   = template.image,
                    name    = f"pneumonia-segmentation-app-{i}",
                    network = "pneumonia-segmentation_default",
                    detach  = True,
                    environment = template.attrs["Config"]["Env"],
                )
        
        elif replicas < current_count:
            sorted_containers = sorted(existing, key=lambda c: c.name)
            for container in sorted_containers[replicas:]:
                container.stop()
                container.remove()
            
        return {"status": f"SCALED app TO {replicas} REPLICAS"}
    except Exception as e:
        logger.error(f"Scale failed: {e}")
        return {"status": "SCALE FAILED"}
        
def _get_compose_project() -> str:
    return os.path.basename(os.getcwd()).lower()

# Swap model
def _swap_model_version(model_file: str = None) -> dict:
    try:
        for attempt in range(3):
        # Switch model
            res = httpx.post(
                f"{APP_URL}/switch-model",
                params={"model_file": model_file},
                timeout=60
            )
            res.raise_for_status()
        
        # Health check after swapping
        health = httpx.get(f"{APP_URL}/health", timeout=10)
        if health.status_code == 200:
            logger.info(f"Model swapped to {model_file}, health OK")
            return {"status": f"SWAPPED TO {model_file.upper()}", "health": "ok"}
        else:
            logger.warning("Model swapped but health check failed")
            return {"status": "SWAPPED BUT UNHEALTHY"}
        
    except Exception as e:
        logger.error(f"Swap failed: {e}")
        return {"status": "SWAP FAILED"}