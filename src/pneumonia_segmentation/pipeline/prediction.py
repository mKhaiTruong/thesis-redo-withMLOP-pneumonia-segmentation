import sys, os
import numpy as np
import cv2
import onnxruntime as ort
from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.constants import *
from pneumonia_segmentation.config import ConfigurationManager
from pneumonia_segmentation.components.data_drift_detector import DataDriftDetector

from core.prometheus_metrics import DRIFT_SCORE, IS_DRIFT

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import hf_hub_download

class PredictionPipeline:
    def __init__(self):
        self.config  = ConfigurationManager().get_evaluation_config()
        drift_config = ConfigurationManager().get_data_drift_config()
        self.session = self._get_onnx_session()
        self._pull_baseline()
        self.data_drift_detector = DataDriftDetector(drift_config)
    
    def _pull_artifact(self):
        target_path = Path("artifacts/best_model_int8.onnx")
        
        if not target_path.exists():
            logging.info(f"Model not found at {target_path}. Downloading from HF...")
            target_path.parent.mkdir(parents=True, exist_ok=True)
        
            hf_hub_download(
                repo_id="bill123mk/pneumonia-seg-weights",
                filename="best_model_int8.onnx",
                local_dir="artifacts",
                local_dir_use_symlinks=False
            )
            logging.info("Download complete.")
        else:
            logging.info("Model already exists locally. Skipping download.")
            
    def _pull_baseline(self):
        target_path = Path("artifacts/data_drift/baseline_distribution.npy")
        
        if not target_path.exists():
            logging.info(f"Baseline not found. Downloading from HF...")
            target_path.parent.mkdir(parents=True, exist_ok=True)
        
            hf_hub_download(
                repo_id="bill123mk/pneumonia-seg-weights",
                filename="baseline_distribution.npy",
                local_dir="artifacts/data_drift",
                local_dir_use_symlinks=False
            )
            logging.info("Download complete.")
        else:
            logging.info("Baseline already exists locally. Skipping download.")
            
    def _get_onnx_session(self):
        try:
            self._pull_artifact()
            
            model_path = "artifacts/best_model_int8.onnx"
            
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            logging.info(f"ONNX model loaded successfully from {model_path}")
            return session
        except Exception as e:
            raise CustomException(e, sys)
            
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        min_val, max_val = gray.min(), gray.max()
        norm = np.clip(gray, min_val, max_val)
        norm = ((norm - min_val) / (max_val - min_val + 1e-5)) * 255
        norm = norm.astype(np.uint8)
        norm = cv2.resize(norm, (self.config.eval.image_size, self.config.eval.image_size))
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def _postprocess(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = cv2.resize(mask, (self.config.eval.image_size, self.config.eval.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return np.concatenate([image, mask], axis=1)

    def _preprocess_for_onnx(self, image, image_size):
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        return image.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        
        image_for_drift = self._preprocess(image)
        
        # --- STAGE 3: DATA DRIFT CHECK ---
        temp_path = Path("temp_inference.png")
        cv2.imwrite(str(temp_path), image_for_drift)
        
        current_feat = self.data_drift_detector.extractor.extract(temp_path)
        baseline_feat = np.load(self.data_drift_detector.config.baseline_dir)
        
        from numpy.linalg import norm
        score = float(norm(baseline_feat - current_feat))
        is_drift = bool(score > self.data_drift_detector.config.metric.drift_threshold)
        
        DRIFT_SCORE.set(score)
        IS_DRIFT.set(int(is_drift))
        drift_result = {
            "drift_score": score,
            "is_drift": is_drift
        }
        
        # --- STAGE 5: INFERENCE ---
        image_rgb = cv2.cvtColor(image_for_drift, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess_for_onnx(image_rgb, self.config.eval.image_size)
        
        ort_inputs = {self.session.get_inputs()[0].name: tensor}
        output = self.session.run(None, ort_inputs)[0]
        
        mask    = (output > 0.5).squeeze().astype(np.uint8)
        result  = self._postprocess(image_for_drift, mask)
        
        if temp_path.exists():
            temp_path.unlink()
        return result, drift_result