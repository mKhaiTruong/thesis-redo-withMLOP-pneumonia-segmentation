from lstm import MetricConfig

METRICS = [
    MetricConfig(
        name  = "cpu",
        query = 'rate(process_cpu_seconds_total{job="app"}[1m]) * 100',
        scale = 100.0
    ),
    MetricConfig(
        name  = "ram",
        query = 'process_resident_memory_bytes{job="app"} / 1024 / 1024',
        scale = 1024.0
    ),
    MetricConfig(
        name  = "latency",
        query = 'histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app", endpoint="/predict"}[5m]))',
        scale = 1.0
    ),
    MetricConfig(
        name  = "drift",
        query = 'inference_drift_score{job="app"}',
        scale = 100.0
    ),
]