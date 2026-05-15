# =============================================================================
# Single source of truth for all Prometheus queries.
# Organized by WHAT is being monitored, not which service uses them.
# =============================================================================



# --- App service metrics (used by LSTM to monitor, Claude to reason) ---------
APP_METRICS = {
    "ram":      'process_resident_memory_bytes{job="app"} / 1024 / 1024',
    "latency":  'histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app", endpoint="/predict"}[5m]))',
    "drift":    'inference_drift_score{job="app"}',
    "requests": 'sum(rate(service_requests_total{job="app"}[1m]))',
}


# --- Self-health metrics (used by Grafana / health checks only) ---------------
LSTM_SELF_METRICS = {
    "ram":      'process_resident_memory_bytes{job="lstm"} / 1024 / 1024',
    "requests": 'sum(rate(service_requests_total{job="lstm"}[1m]))',
}

DQN_SELF_METRICS = {
    "ram":      'process_resident_memory_bytes{job="dqn"} / 1024 / 1024',
    "requests": 'sum(rate(service_requests_total{job="dqn"}[1m]))',
}


# --- Backward-compat flat dict (used by claude_arch, grafana, dashboard) ------
QUERIES = {
    "app_latency":   APP_METRICS["latency"],
    "app_ram":       APP_METRICS["ram"],
    "app_drift":     APP_METRICS["drift"],
    "app_requests":  APP_METRICS["requests"],
    "lstm_ram":      LSTM_SELF_METRICS["ram"],
    "lstm_requests": LSTM_SELF_METRICS["requests"],
    "dqn_ram":       DQN_SELF_METRICS["ram"],
    "dqn_requests":  DQN_SELF_METRICS["requests"],
}