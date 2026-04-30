import torch
import numpy as np

from core.logging import logger
from lstm import LSTM_Predictor_Config
from lstm.model import LSTM_Predictor
from lstm.utils.query_prometheus import _query_prometheus
from lstm.utils.metrics import METRICS

class LSTM_Predictor_:
    def __init__(self, config: LSTM_Predictor_Config, 
                 prometheus_url: str = "http://localhost:9090"):
        self.config         = config
        self.model_path     = str(self.config.root_dir / "lstm_model.pth")
        self.prometheus_url = prometheus_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model()
    
    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.input_steps  = checkpoint["input_steps"]
        self.output_steps = checkpoint["output_steps"]
        self.mean = checkpoint["mean"]
        self.std  = checkpoint["std"]
        
        hidden_size = checkpoint.get("hidden_size", 64)
        num_layers  = checkpoint.get("num_layers", 2)
        
        self.model = LSTM_Predictor(
            input_size  = self.config.lstm_params.input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            output_steps = self.output_steps
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("LSTM model loaded")
        
        return self.model

    def predict(self) -> dict:
        all_metrics = self._query_all()
        metrics     = torch.tensor(
            list(zip(*all_metrics)),
            dtype   = torch.float32
        ).unsqueeze(0).to(self.device)
        
        data = torch.clamp(metrics, 0.0, 1.0)
        logger.info(f"Normalized input sample: {data[0, -1, :].tolist()}")
        with torch.no_grad():
            pred = self.model(data)
        
        # Denormalize
        pred = torch.clamp(pred, 0.0, 1.0)
        pred = pred.cpu().numpy()
        
        if not np.isfinite(pred).all():
            logger.warning("LSTM predicted NaN/inf — returning zeros")
            self.model = self._load_model()
            return {
                "predicted_cpu":     [0.0] * self.output_steps,
                "predicted_ram":     [0.0] * self.output_steps,
                "predicted_latency": [0.0] * self.output_steps,
                "predicted_drift":   [0.0] * self.output_steps,
                "steps":             self.output_steps
            }
        
        return {
            "predicted_cpu":     pred[0, :, 0].tolist(),
            "predicted_ram":     pred[0, :, 1].tolist(),
            "predicted_latency": pred[0, :, 2].tolist(),
            "predicted_drift":   pred[0, :, 3].tolist(),
            "steps":             self.output_steps
        }
        
    def get_current_state(self) -> dict:
        all_metrics = self._query_all()
        return {
            f"current_{m.name}": [vals[-1] if np.isfinite(vals[-1]) else 0.0]
            for m, vals in zip(METRICS, all_metrics)
        }
        
    def _query_all(self) -> list[list[float]]:
        return [
            _query_prometheus(
                query = m.query, 
                scale = m.scale, 
                input_steps     = self.output_steps, 
                prometheus_url  = self.prometheus_url)
            for m in METRICS
        ]