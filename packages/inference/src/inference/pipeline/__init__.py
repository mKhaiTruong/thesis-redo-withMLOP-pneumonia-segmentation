import numpy as np
from inference.config import ConfigurationManager
from inference.components.model_loader import ModelLoader
from inference.components.preprocessor import Preprocessor
from inference.components.postprocessor import Postprocessor
from inference.components.drift_checker import DriftChecker

class PredictionPipeline:
    def __init__(self):
        config        = ConfigurationManager()
        eval_config   = config.get_evaluation_config()
        drift_config  = config.get_data_drift_config()

        self.model  = ModelLoader(eval_config)
        self.pre    = Preprocessor(eval_config)
        self.post   = Postprocessor(eval_config)
        self.drift  = DriftChecker(drift_config)

    def predict(self, image: np.ndarray):
        processed    = self.pre.process(image)
        drift_result = self.drift.check(processed)
        tensor       = self.pre.to_tensor(processed)
        output       = self.model.run(tensor)
        result       = self.post.process(processed, output)
        return result, drift_result

    def reload(self):
        self.model.reload()