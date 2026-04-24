from pathlib import Path
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from core.logging import logger
from core.exception import CustomException
import sys

class ModelLoader:
    HF_REPO    = "bill123mk/pneumonia-seg-weights"
    MODEL_FILE = "best_model_int8.onnx"
    LOCAL_PATH = Path("artifacts/best_model_int8.onnx")
    
    def __init__(self, config):
        self.config  = config
        self.session = self._load()
        
    def _pull(self):
        if not self.LOCAL_PATH.exists():
            logger.info("Model not found locally. Downloading from HF...")
            self.LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(
                repo_id     = self.HF_REPO,
                filename    = self.MODEL_FILE,
                local_dir   = "artifacts",
                local_dir_use_symlinks=False
            )
            logger.info("Download complete.")
        else:
            logger.info("Model already exists locally. Skipping download.")
    
    def _load(self):
        try:
            self._pull()
            session = ort.InferenceSession(
                str(self.LOCAL_PATH),
                providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX model loaded from {self.LOCAL_PATH}")
            return session
        except Exception as e:
            raise CustomException(e, sys)

    def run(self, tensor):
        inputs = {self.session.get_inputs()[0].name: tensor}
        return self.session.run(None, inputs)[0]
    
    def reload(self):
        self.LOCAL_PATH.unlink(missing_ok=True)
        self.session = self._load()
        logger.info("Model reloaded.")