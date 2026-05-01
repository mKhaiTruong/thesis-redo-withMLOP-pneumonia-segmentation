import httpx

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin")

def create_dashboard(panels: list) -> dict:
    return {
        "dashboard": {
            "title": "MLOps — Pneumonia Segmentation",
            "uid": "pneumonia-mlops",
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels
        },
        "overwrite": True,
        "folderId": 0,
    }

def push_dashboard(dashboard: dict):
    r = httpx.post(
        f"{GRAFANA_URL}/api/dashboards/db",
        json=dashboard,
        auth=AUTH,
        timeout=30
    )
    r.raise_for_status()
    print(f"Dashboard created: {r.json()['url']}")
    
def stat_panel(title: str, expr: str, unit: str, thresholds: list, grid: dict) -> dict:
    return {
        "type":         "stat",
        "title":        title,
        "gridPos":      grid,
        "datasource":   {"type": "prometheus", "uid": "prometheus"},
        "options":      {
            "colorMode": "background",
            "reduceOptions": {"calcs": ["lastNotNull"]}
        },
        "targets":      [{"expr": expr, "legendFormat": title}],
        "fieldConfig":  {
            "defaults": {
                "unit":         unit,
                "color":        {"mode": "thresholds"},
                "thresholds":   {
                    "mode": "absolute",
                    "steps": thresholds
                }
            }
        }
    }
    
def timeseries_panel(title: str, targets: list, unit: str, grid: dict) -> dict:
    return {
        "type":         "timeseries",
        "title":        title,
        "gridPos":      grid,
        "datasource":   {"type": "prometheus", "uid": "prometheus"},
        "options":      {
            "tooltip":  {"mode": "multi"},
            "legend":   {"displayMode": "list", "placement": "bottom"}
        },
        "targets":      [
            {"expr": expr, "legendFormat": legend}
            for expr, legend in targets
        ],
        "fieldConfig":  {
            "defaults": {"unit": unit}
        },
    }