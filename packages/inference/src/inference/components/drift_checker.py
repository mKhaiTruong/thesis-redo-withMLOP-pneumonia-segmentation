import sys
import numpy as np
import cv2
from pathlib import Path
from numpy.linalg import norm
from huggingface_hub import hf_hub_download
from core.logging import logger
from core.exception import CustomException
from core.prometheus_metrics import DRIFT_SCORE, IS_DRIFT

class DriftChecker:
    HF_REPO       = "bill123mk/pneumonia-seg-weights"
    BASELINE_FILE = "baseline_distribution.npy"
    LOCAL_PATH    = Path("artifacts/data_drift/baseline_distribution.npy")

    def __init__(self, config):
        self.config    = config
        self.extractor = self._init_extractor()
        self._pull_baseline()

    def _init_extractor(self):
        from data_drift.components import DataDriftDetector
        return DataDriftDetector(self.config).extractor

    def _pull_baseline(self):
        if not self.LOCAL_PATH.exists():
            logger.info("Baseline not found. Downloading from HF...")
            self.LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=self.HF_REPO,
                filename=self.BASELINE_FILE,
                local_dir="artifacts/data_drift",
                local_dir_use_symlinks=False
            )

    def check(self, image: np.ndarray) -> dict:
        try:
            temp_path = Path("temp_inference.png")
            cv2.imwrite(str(temp_path), image)

            current_feat  = self.extractor.extract(temp_path)
            baseline_feat = np.load(self.LOCAL_PATH)

            score    = float(norm(baseline_feat - current_feat))
            is_drift = bool(score > self.config.metric.drift_threshold)

            DRIFT_SCORE.set(score)
            IS_DRIFT.set(int(is_drift))

            return {"drift_score": score, "is_drift": is_drift}
        except Exception as e:
            raise CustomException(e, sys)
        finally:
            if temp_path.exists():
                temp_path.unlink()