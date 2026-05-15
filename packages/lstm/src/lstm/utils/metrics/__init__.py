from core.metrics_queries import LSTM_QUERIES
from lstm import MetricConfig

METRICS = [
    # MetricConfig(
    #     name  = "cpu",
    #     query = 'rate(process_cpu_seconds_total{job="app"}[1m]) * 100',
    #     scale = 100.0
    # ),
    MetricConfig(
        name  = "ram",
        query = LSTM_QUERIES['lstm_ram'],
        scale = 1024.0
    ),
    MetricConfig(
        name  = "latency",
        query = LSTM_QUERIES['lstm_latency'],
        scale = 1.0
    ),
    MetricConfig(
        name  = "drift",
        query = LSTM_QUERIES['lstm_drift'],
        scale = 100.0
    ),
    MetricConfig(
        name  = "requests",
        query = LSTM_QUERIES['lstm_requests'],
        scale = 100.0
    ),
]