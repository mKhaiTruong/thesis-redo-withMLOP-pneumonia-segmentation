import time, requests, math, os
from prometheus_client import Gauge, push_to_gateway, CollectorRegistry

IMAGE_PATH = r"D:\Deep_Learning_Object_Detection\MLOPs\pneumonia-segmentation\artifacts\data_transformation\infer\img\scan_00002.png"
URL = "http://localhost:7860/predict"

registry = CollectorRegistry()
SYNTHETIC_DRIFT = Gauge(
    'synthetic_drift_score', 
    'Drift score generated for LSTM training', 
    registry=registry
)

step = 0
while True:
    try:
        with open(IMAGE_PATH, 'rb') as f:
            requests.post(URL, files={"file": f})
        
        val = 0.5 + 0.3 * math.sin(step)
        SYNTHETIC_DRIFT.set(val)
        print(f"Simulated Drift: {val:.2f}")
        
        step += 0.2
        time.sleep(2)
    except Exception as e:
        print(f"Req: Failed - {e} 🔥")