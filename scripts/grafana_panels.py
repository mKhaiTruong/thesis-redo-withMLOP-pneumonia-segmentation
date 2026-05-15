from setup_grafana import *
from core.metrics_queries import QUERIES

# Row 1 — stat cards
PANELS = [
    stat_panel(
        title="Latency P95", unit="s",
        expr=QUERIES["app_latency"],
        grid={"h": 4, "w": 6, "x": 0, "y": 0},
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 0.5},
            {"color": "red", "value": 1.0}
        ],
    ),
    stat_panel(
        title="Req/s", unit="reqps",
        expr=QUERIES["app_requests"],
        grid={"h": 4, "w": 6, "x": 6, "y": 0},
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 50},
            {"color": "red", "value": 100}
        ],
    ),
    stat_panel(
        title="RAM MB", unit="decmbytes",
        expr=QUERIES["app_ram"],
        grid={"h": 4, "w": 6, "x": 18, "y": 0},
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 512},
            {"color": "red", "value": 1024}
        ],
    ),
    stat_panel(
        title="Drift Score", unit="short",
        expr=QUERIES["app_drift"],
        grid={"h": 4, "w": 6, "x": 12, "y": 0},
        thresholds=[
            {"color": "green", "value": None},
            {"color": "yellow", "value": 40},
            {"color": "red", "value": 60}
        ],
    ),
]

# Row 2 — timeseries
PANELS.append(timeseries_panel(
    title="Latency over time", unit="s",
    targets=[(QUERIES["app_latency"], "p95")],
    grid={"h": 8, "w": 12, "x": 0, "y": 4}
))
PANELS.append(timeseries_panel(
    title="Drift over time", unit="short",
    targets=[(QUERIES["app_drift"], "drift")],
    grid={"h": 8, "w": 12, "x": 12, "y": 4}
))

# Row 3 — all services
PANELS.extend([
    timeseries_panel(
        title="RAM — all services", unit="decmbytes",
        targets=[
            (QUERIES["app_ram"],  "app"),
            (QUERIES["lstm_ram"], "lstm"),
            (QUERIES["dqn_ram"],  "dqn"),
        ],
        grid={"h": 8, "w": 12, "x": 0, "y": 12}
    ),
    timeseries_panel(
        title="Requests/s — all services", unit="reqps",
        targets=[
            (QUERIES["app_requests"],  "app"),
            (QUERIES["lstm_requests"], "lstm"),
            (QUERIES["dqn_requests"],  "dqn"),
        ],
        grid={"h": 8, "w": 12, "x": 12, "y": 12}
    ),
])

if __name__ == "__main__":
    setup_datasource()
    dashboard = create_dashboard(PANELS)
    push_dashboard(dashboard)