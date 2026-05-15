import torch
import numpy as np

from core.logging import logger
from lstm import LSTM_Predictor_Config
from lstm.model import LSTM_Predictor
from lstm.utils.query_prometheus import _query_prometheus
from lstm import METRICS, METRIC_NAMES

class LSTM_Predictor_:
    def __init__(self, config: LSTM_Predictor_Config,  prometheus_url: str):
        self.config         = config
        self.model_path     = str(self.config.model_dir)
        self.prometheus_url = prometheus_url
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loaded from checkpoint — set by _load_model()
        self.model:        LSTM_Predictor | None = None
        self.mean:         torch.Tensor  | None = None
        self.std:          torch.Tensor  | None = None
        self.input_steps:  int = 0
        self.output_steps: int = 0
    
        self.model  = self._load_model()
        
        
    def _load_model(self) -> LSTM_Predictor:
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Sanity check — ensure checkpoint was trained with same features
        ckpt_names = checkpoint.get("metric_names")
        if ckpt_names and ckpt_names != METRIC_NAMES:
            raise ValueError(
                f"Checkpoint metric_names={ckpt_names} "
                f"does not match current METRIC_NAMES={METRIC_NAMES}. "
                f"Retrain the model."
            )
        
        self.input_steps  = checkpoint["input_steps"]
        self.output_steps = checkpoint["output_steps"]
        
        # Keep mean/std on CPU for easy arithmetic, move to device when needed
        self.mean = checkpoint["mean"].to(self.device)
        self.std  = checkpoint["std"].to(self.device)
        
        model = LSTM_Predictor(
            input_size  = self.config.lstm_params.input_size,
            hidden_size = checkpoint.get("hidden_size", 64),
            num_layers  = checkpoint.get("num_layers", 2),
            output_steps = self.output_steps
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        logger.info(
            f"LSTM model loaded"
            f"input_steps={self.input_steps} | output_steps={self.output_steps} | "
            f"features={METRIC_NAMES}"
        )
        return model


    def predict(self) -> dict:
        all_metrics = self._query_all()
        logger.info(f"Raw input — ram={all_metrics[0][-3:]}, latency={all_metrics[1][-3:]}, drift={all_metrics[2][-3:]}, requests={all_metrics[3][-3:]}")
        
        metrics_tensor = torch.tensor(
            list(zip(*all_metrics)), dtype = torch.float32
        ).unsqueeze(0).to(self.device)  # [1, input_steps, num_features]
        
        # Normalize — must mirror _prepare_loader() in trainer.py
        data = (metrics_tensor - self.mean) / (self.std + 1e-8)
        
        with torch.no_grad():
            pred = self.model(data)     # [1, output_steps, num_features]
        
        # Denormalize back to real unit
        pred = pred * self.std + self.mean
        pred = pred.cpu().numpy()       # [1, output_steps, num_features]
        
        if not np.isfinite(pred).all():
            logger.warning("LSTM predicted NaN/inf — reloading model, returning zeros")
            self.model = self._load_model()
            return self._zero_prediction()

        # current_state: last observed value for each metric (real units)
        current = {
            f"current_{m.name}": float(all_metrics[i][-1])
            if np.isfinite(all_metrics[i][-1]) else 0.0
            for i, m in enumerate(METRICS)
        }
        
        # predictions: index-driven, no hardcoded column positions
        predictions = {
            f"predicted_{name}": pred[0, :, i].tolist()
            for i, name in enumerate(METRIC_NAMES)
        }
        
        return {
            **current, 
            **predictions,
            "steps":           self.output_steps,
            "horizon_minutes": self.output_steps,  # 1 step ≈ 1 min (15s*4 = 1min)
        }
        
        
    def get_current_state(self) -> dict:
        all_metrics = self._query_all()
        return {
            f"current_{m.name}": float(all_metrics[i][-1])
            if np.isfinite(all_metrics[i][-1]) else 0.0
            for i, m in enumerate(METRICS)
        }
        
    def _query_all(self) -> list[list[float]]:
        return [
            _query_prometheus(
                query           = m.query, 
                scale           = m.scale, 
                input_steps     = self.input_steps, 
                prometheus_url  = self.prometheus_url
            )
            for m in METRICS
        ]
    
    def _zero_prediction(self) -> dict:
        return {
            **{f"current_{name}": 0.0 for name in METRIC_NAMES},
            **{f"predicted_{name}": [0.0] * self.output_steps for name in METRIC_NAMES},
            "steps":           self.output_steps,
            "horizon_minutes": self.output_steps,
        }
            