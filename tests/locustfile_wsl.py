import io
from locust import HttpUser, task, between

class MLOpsUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://pneumonia.local"

    @task(3)
    def predict(self):
        self.client.post("/predict-mock", params={"latency": 2.5})

    @task(1)
    def health_check(self):
        self.client.get("/health")