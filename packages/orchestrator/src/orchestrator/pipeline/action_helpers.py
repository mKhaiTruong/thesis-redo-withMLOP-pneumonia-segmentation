import os, httpx, docker
from docker_composer import DockerCompose
from core.logging import logger
from orchestrator import APP_URL

_docker_client = docker.from_env()

# Scale app
def _scale_app(replicas: int) -> dict: 
    try:
        project = _get_compose_project()
        DockerCompose(project_name=project).scale(
            **{"app": replicas}
        ).call()
            
        return {"status": f"SCALED app TO {replicas} REPLICAS"}
    except Exception as e:
        logger.error(f"Scale failed: {e}")
        return {"status": "SCALE FAILED"}
        
def _get_compose_project() -> str:
    return os.path.basename(os.getcwd()).lower()

# Swap model
def _swap_model_version(model_type: str = None) -> dict:
    try:
        # Switch model
        res = httpx.post(
            f"{APP_URL}/switch-model",
            params={"model_type": model_type},
            timeout=10
        )
        res.raise_for_status()
        
        # Health check after swapping
        health = httpx.get(f"{APP_URL}/health", timeout=10)
        if health.status_code == 200:
            logger.info(f"Model swapped to {model_type}, health OK")
            return {"status": f"SWAPPED TO {model_type.upper()}", "health": "ok"}
        else:
            logger.warning("Model swapped but health check failed")
            return {"status": "SWAPPED BUT UNHEALTHY"}
        
    except Exception as e:
        logger.error(f"Swap failed: {e}")
        return {"status": "SWAP FAILED"}