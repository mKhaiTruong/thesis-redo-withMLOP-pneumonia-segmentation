import requests
import time
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:7860/predict"
IMAGE_PATH = r'artifacts/data_transformation/infer/img/scan_00008.png'

def send_request(i):
    with open(IMAGE_PATH, 'rb') as f:
        files = {'file': ('scan.png', f, 'image/png')}
        
        try:
            r = requests.post(URL, files=files)
            print(f"Req {i}: Status {r.status_code} - Drift: {r.headers.get('X-Drift-Status')}")
        except Exception as e:
            print(f"Req {i}: Failed - {e}")

with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(send_request, range(50))
