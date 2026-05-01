import io
from locust import HttpUser, task, between

class MLOpsUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:7860"
    
    def on_start(self):
        img_path = r"D:\Deep_Learning_Object_Detection\MLOPs\pneumonia-segmentation\artifacts\data_transformation\infer\img\scan_00002.png"
        with open(img_path, "rb") as f:
            self.img_bytes = f.read()
    
    @task(3)
    def predict(self):
        self.client.post("/predict", files={
            "file": ("scan_00002.png", io.BytesIO(self.img_bytes), "image/png")
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")