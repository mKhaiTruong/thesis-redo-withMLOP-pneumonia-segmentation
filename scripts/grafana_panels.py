from setup_grafana import *

# Row 1 — stat cards
PANELS = [
    stat_panel(
        title="Latency P95", 
        unit="s",
        expr='histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app", endpoint="/predict"}[5m]))',
        grid={"h": 4, "w": 6, "x": 0, "y": 0},
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.5},
            {"color": "red", "value": 1.0}
        ],
    ),
    stat_panel(
        title="CPU %", unit="percent",
        expr='rate(process_cpu_seconds_total{job="app"}[1m]) * 100',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "red", "value": 70}
        ],
        grid={"h": 4, "w": 6, "x": 6, "y": 0}
    ),
    stat_panel(
        title="RAM MB", unit="decmbytes",
        expr='process_resident_memory_bytes{job="app"} / 1024 / 1024',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 512},
            {"color": "red", "value": 1024}
        ],
        grid={"h": 4, "w": 6, "x": 18, "y": 0}
    ),
    stat_panel(
        title="Drift score", unit="short",
        expr='inference_drift_score{job="app"}',
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 40},
            {"color": "red", "value": 60}
        ],
        grid={"h": 4, "w": 6, "x": 12, "y": 0}
    ),
]

# Row 2 — timeseries
PANELS.append(
    timeseries_panel(
        title   = "Latency over time", unit="s",
        targets = [
            ('histogram_quantile(0.95, rate(service_request_latency_seconds_bucket{job="app", endpoint="/predict"}[5m]))', "p95"),
        ],
        grid = {"h": 8, "w": 12, "x": 0, "y": 4}
    )
)
PANELS.append(
    timeseries_panel(
        title   = "Drift over time", unit="short",
        targets = [
            ('inference_drift_score{job="app"}', "drift"),
        ],
        grid = {"h": 8, "w": 12, "x": 12, "y": 4}
    )
)

# Row 3 — all services
PANELS.extend([
    timeseries_panel(
        title   = "CPU — all services", unit="percent",
        targets = [
            ('rate(process_cpu_seconds_total{job="app"}[1m]) * 100', "app"),
            ('rate(process_cpu_seconds_total{job="lstm"}[1m]) * 100', "lstm"),
            ('rate(process_cpu_seconds_total{job="dqn"}[1m]) * 100', "dqn"),
        ],
        grid = {"h": 8, "w": 12, "x": 0, "y": 12}
    ),
    timeseries_panel(
        title   = "RAM — all services", unit="decmbytes",
        targets = [
            ('process_resident_memory_bytes{job="app"} / 1024 / 1024', "app"),
            ('process_resident_memory_bytes{job="lstm"} / 1024 / 1024', "lstm"),
            ('process_resident_memory_bytes{job="dqn"} / 1024 / 1024', "dqn"),
        ],
        grid = {"h": 8, "w": 12, "x": 12, "y": 12}
    ),
])

if __name__ == "__main__":
    dashboard = create_dashboard(PANELS)
    push_dashboard(dashboard)