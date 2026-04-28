import httpx

BASE = "http://localhost"

checks = {
    "app health":               f"{BASE}:7860/health",
    "lstm health":              f"{BASE}:7868/health", 
    "dqn health":               f"{BASE}:7869/health",
    "ai_manager health":        f"{BASE}:7870/health",
    "claude_validation health": f"{BASE}:7871/health",
    "prometheus":               f"{BASE}:9090/-/healthy",
    "grafana":                  f"{BASE}:3000/api/health",
}

print("=== Service Health ===")
for name, url in checks.items():
    try:
        r = httpx.get(url, timeout=3)
        print(f"{'✅' if r.status_code == 200 else '❌'} {name}: {r.status_code}")
    except Exception as e:
        print(f"❌ {name}: {e}")
        
print("\n=== Prometheus Metrics ===")
metrics = [
    "inference_drift_score",
    "service_request_latency_seconds_count",
    "process_cpu_seconds_total",
    "process_resident_memory_bytes",
]
for m in metrics:
    r = httpx.get(f"{BASE}:9090/api/v1/query", params={"query": m})
    result = r.json()["data"]["result"]
    val = result[0]["value"][1] if result else "NO DATA"
    print(f"{'✅' if result else '❌'} {m}: {val}")