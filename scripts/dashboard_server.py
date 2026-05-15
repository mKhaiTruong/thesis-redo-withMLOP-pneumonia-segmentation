import subprocess, asyncio, httpx
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_URL    = "http://pneumonia.local"
CLAUDE_URL  = "http://pneumonia.local/claude-architecture"
LOCUST_FILE = "tests/locustfile_wsl.py"

locust_process = None

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    return HTMLResponse(content=(Path(__file__).parent / "dashboard.html").read_text())

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/scenario/{scenario}")
async def trigger_scenario(scenario: str):
    global locust_process
    configs = {
        "high_load":   {"drift": 10,  "users": 200, "rate": 20},
        "model_drift": {"drift": 75,  "users": 20,  "rate": 2},
        "degraded":    {"drift": 15,  "users": 50,  "rate": 5},
    }
    cfg = configs.get(scenario)
    if not cfg:
        return {"error": f"Unknown scenario: {scenario}"}

    if locust_process and locust_process.poll() is None:
        subprocess.run(["pkill", "-f", "locust"], capture_output=True)
        locust_process.wait()

    try:
        r=httpx.post(f"{BASE_URL}/debug/set-drift?score={cfg['drift']}", timeout=5)
        print(f"set-drift response: {r.status_code} {r.text}")
    except Exception:
        pass

    locust_process = subprocess.Popen([
        "locust", "-f", LOCUST_FILE,
        "--headless", "-u", str(cfg["users"]), "-r", str(cfg["rate"]),
        "--run-time", "1m"
    ])

    return {
        "scenario":     scenario,
        "drift_set":    cfg["drift"],
        "locust_users": cfg["users"],
    }

@app.post("/api/reset")
async def reset():
    subprocess.run(["pkill", "-f", "locust"], capture_output=True)
    await asyncio.sleep(0.5)
    try:
        httpx.post(f"{BASE_URL}/debug/set-drift?score=0", timeout=5)
        httpx.post(f"{BASE_URL}/orchestrator/execute/scale_in_service", timeout=10)
    except Exception:
        pass
    subprocess.run(["kubectl", "scale", "deployment", "app", "--replicas=1"], capture_output=True)
    await asyncio.sleep(5)
    subprocess.run(["kubectl", "rollout", "restart", "deployment/app"], capture_output=True)
    
    await asyncio.sleep(15)
    try:
        httpx.post(f"{BASE_URL}/predict-mock?latency=0.1", timeout=5)
    except Exception:
        pass
    
    return {"status": "reset"}

@app.post("/api/decision")
async def get_decision():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{CLAUDE_URL}/run-claude-architecture", timeout=60)
            return r.json()
    except Exception as e:
        return {"action": "error", "reasoning": str(e), "confidence": 0, "evolution_needed": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)