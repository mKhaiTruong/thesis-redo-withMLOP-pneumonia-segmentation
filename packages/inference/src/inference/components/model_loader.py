from pathlib import Path
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from core.logging import logger
from core.exception import CustomException
import sys

class ModelLoader:
    HF_REPO    = "bill123mk/pneumonia-seg-weights"
    
    def __init__(self, config, model_file = f"best_model_int8.onnx"):
        self.config  = config
        self.model_file = model_file
        self.local_path = Path(f"artifacts/{model_file}")
        self.session = self._load()
        
    def _pull(self):
        if not self.local_path.exists():
            logger.info("Model not found locally. Downloading from HF...")
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(
                repo_id     = self.HF_REPO,
                filename    = self.model_file,
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
                str(self.local_path),
                providers=["CPUExecutionProvider"]
            )
            logger.info(f"ONNX model loaded from {self.local_path}")
            return session
        except Exception as e:
            raise CustomException(e, sys)

    def run(self, tensor):
        inputs = {self.session.get_inputs()[0].name: tensor}
        return self.session.run(None, inputs)[0]
    
    def reload(self):
        self.local_path.unlink(missing_ok=True)
        self.session = self._load()
        logger.info("Model reloaded.")