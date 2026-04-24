import cv2
import numpy as np

class Preprocessor:
    def __init__(self, config):
        self.image_size = config.eval.image_size

    def process(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        min_val, max_val = gray.min(), gray.max()
        norm = np.clip(gray, min_val, max_val)
        norm = ((norm - min_val) / (max_val - min_val + 1e-5)) * 255
        norm = norm.astype(np.uint8)
        norm = cv2.resize(norm, (self.image_size, self.image_size))
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_f   = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_f   = image_f.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image_f = (image_f - mean) / std
        return image_f.transpose(2, 0, 1)[np.newaxis].astype(np.float32)