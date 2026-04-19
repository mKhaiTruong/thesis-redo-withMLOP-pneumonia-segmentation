import numpy as np
from numpy.linalg import norm
from pathlib import Path

from core.logging import logger
from data_drift import DataDriftConfig
from core.utils.data_drift_helpers import FeatureExtractor

class DataDriftDetector:
    def __init__(self, config: DataDriftConfig):
        self.config = config
        self._extractor = None
    
    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = FeatureExtractor(
                model_name=self.config.metric.model_name, 
                device="cpu"
            )
        return self._extractor
    
    def _get_distribution(self, img_paths: list[Path]) -> np.ndarray:
        logger.info(f"Extracting features from {len(img_paths)} images...")
        all_features = [self.extractor.extract(path) for path in img_paths]

        all_features = np.array(all_features)
        return np.mean(all_features, axis=0)
    
    def run(self) -> dict:
        img_paths = sorted(list(self.config.origin_data_source.rglob("*.png")))
        if self.config.metric.max_samples and len(img_paths) > self.config.metric.max_samples:
            rng = np.random.default_rng(self.config.metric.seed)
            indices = rng.choice(len(img_paths), self.config.metric.max_samples, replace=False)
            img_paths = [img_paths[i] for i in sorted(indices)]
            
        current_avg_feat = self._get_distribution(img_paths)
        if not self.config.baseline_dir.exists():
            np.save(self.config.baseline_dir, current_avg_feat)
            return {"status": "Baseline Created"}

        baseline_avg_feat = np.load(self.config.baseline_dir)
        score = norm(baseline_avg_feat - current_avg_feat)
        
        return {
            "drift_score": float(score),
            "is_drift": bool(score > self.config.metric.drift_threshold),
            "n_images": len(img_paths),
            "threshold": float(self.config.metric.drift_threshold)
        }