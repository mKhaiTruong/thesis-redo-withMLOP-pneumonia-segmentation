import torch
import httpx
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.logging import logger
from lstm import LSTM_Predictor_Config
from lstm.model import LSTM_Predictor

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
    
    def _query_prometheus(self, query: str) -> list[float]:
        end     = datetime.now(timezone.utc)
        start   = end - timedelta(minutes=self.input_steps * 15 // 60 + 5)
        
        response = httpx.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start.isoformat(),
                "end":   end.isoformat(),
                "step": "15s"
            }
        )
        
        body = response.json()
        
        # --- debug ---
        if body.get("status") != "success":
            logger.warning(f"Prometheus query failed | query={query} | response={body}")
            return [0.0] * self.input_steps
        
        result = body["data"]["result"]
        if not result:
            logger.warning(f"Prometheus empty result | query={query}")
            return [0.0] * self.input_steps
        
        values = [float(v[1]) for v in result[0]["values"]]
        if len(values) < self.input_steps:
            values = [values[0]] * (self.input_steps - len(values)) + values
        return values[-self.input_steps:]


    def predict(self) -> dict:
        cpu     = self._query_prometheus('rate(process_cpu_seconds_total{job="app"}[1m]) * 100')
        ram     = self._query_prometheus('process_resident_memory_bytes{job="app"} / 1024 / 1024')
        latency = self._query_prometheus('histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app"}[5m]))')
        drift   = self._query_prometheus('inference_drift_score{job="app"}')
        
        metrics = torch.tensor(
            list(zip(cpu, ram, latency, drift)),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        data = (metrics - self.mean.to(self.device)) / (self.std.to(self.device) + 1e-8)
        with torch.no_grad():
            pred = self.model(data)
        
        # Denormalize
        pred = pred * self.std.to(self.device) + self.mean.to(self.device)
        pred = pred.cpu().numpy()
        
        if not np.isfinite(pred).all():
            logger.warning("LSTM predicted NaN/inf — returning zeros")
            pred = np.zeros_like(pred)
        
        return {
            "predicted_cpu":     pred[0, :, 0].tolist(),
            "predicted_ram":     pred[0, :, 1].tolist(),
            "predicted_latency": pred[0, :, 2].tolist(),
            "predicted_drift":   pred[0, :, 3].tolist(),
            "steps":             self.output_steps
        }
        
    def get_current_state(self) -> dict:
        def safe_last(values: list[float]) -> float:
            val = values[-1]
            return val if np.isfinite(val) else 0.0
    
        return {
            "current_cpu":     [safe_last(self._query_prometheus('rate(process_cpu_seconds_total{job="app"}[1m]) * 100'))],
            "current_ram":     [safe_last(self._query_prometheus('process_resident_memory_bytes{job="app"} / 1024 / 1024'))],
            "current_latency": [safe_last(self._query_prometheus('histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app"}[5m]))'))],
            "current_drift":   [safe_last(self._query_prometheus('inference_drift_score{job="app"}'))],
        }