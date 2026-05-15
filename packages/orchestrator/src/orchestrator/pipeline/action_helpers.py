import os, httpx
from kubernetes import client, config as k8s_config
from core.logging import logger
from orchestrator import APP_URL

# Scale app
def _scale_app(replicas: int) -> dict: 
    try:
        try:
            k8s_config.load_incluster_config()
        except:
            k8s_config.load_kube_config()
        
         
        deployments = ["app-edge", "app"]
        apps_v1 = client.AppsV1Api()
        scaled = None
        
        for d in deployments:
            try:
                apps_v1.patch_namespaced_deployment_scale(
                    name = d,
                    namespace = "default",
                    body = {"spec": {"replicas": replicas}}
                )
                scaled = d
                break
            except Exception:
                continue
            
        if scaled:
            return {"status": f"SCALED {scaled} TO {replicas} REPLICAS"}
        
        return {"status": "SCALE FAILED - no deployment found"}
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