from dataclasses import dataclass
from pathlib import Path
from core.metrics_queries import APP_METRICS

@dataclass(frozen=True)
class LSTM_Params_Config:
    input_size:  int
    hidden_size: int
    num_layers:  int
    input_steps: int
    output_steps: int
    epochs:     int
    batch_size: int
    lr:         float

@dataclass(frozen=True)
class LSTM_Predictor_Config:
    root_dir:       Path
    model_dir:      Path
    lstm_params:    LSTM_Params_Config

@dataclass
class MetricConfig:
    name:    str
    query:   str
    scale:   float = 1.0

METRICS: list[MetricConfig] = [
    MetricConfig(name="ram",      query=APP_METRICS["ram"]),       # MB
    MetricConfig(name="latency",  query=APP_METRICS["latency"]),   # seconds (P95)
    MetricConfig(name="drift",    query=APP_METRICS["drift"]),     # score 0–1
    MetricConfig(name="requests", query=APP_METRICS["requests"]),  # req/s
]

METRIC_NAMES: list[str] = [m.name for m in METRICS]