from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

class FeatureExtractor:
    def __init__(self, model_name: str, device: str):
        # Download ResNet50 ONNX từ ONNX Model Zoo
        self.session = ort.InferenceSession(
            model_name,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, img_path: Path) -> np.ndarray:
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def extract(self, img_path: Path) -> np.ndarray:
        tensor = self._preprocess(img_path)
        feat = self.session.run(None, {self.input_name: tensor})
        return feat[0].squeeze()