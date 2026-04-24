import cv2
import numpy as np

class Postprocessor:
    def __init__(self, config):
        self.image_size = config.eval.image_size

    def process(self, image: np.ndarray, output: np.ndarray) -> np.ndarray:
        mask = (output > 0.5).squeeze().astype(np.uint8)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return np.concatenate([image, mask], axis=1)