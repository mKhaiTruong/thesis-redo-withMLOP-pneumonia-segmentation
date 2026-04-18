import torch
from core.logging import logger

def get_device():
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        logger.warning("No GPU detected — falling back to CPU. Training will be slower.")
        return torch.device("cpu")