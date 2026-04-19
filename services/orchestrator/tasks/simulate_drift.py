import time
import random
import math
from prometheus_client import Gauge, start_http_server

DRIFT_SCORE = Gauge("inference_drift_score", "Synthetic drift score for testing")

if __name__ == "__main__":
    start_http_server(8000)
    step = 0
    while True:
        val = 0.5 + 0.3 * math.sin(step) + random.uniform(-0.05, 0.05)
        DRIFT_SCORE.set(val)
        print(f"Bơm data: {val:.4f}")
        step += 0.1
        time.sleep(5)