from locust import HttpUser, task, between

class MLOpsUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:7860"
    
    @task(3)
    def predict(self):
        import base64
        from pathlib import Path
        
        img_path = r"D:\Deep_Learning_Object_Detection\MLOPs\pneumonia-segmentation\artifacts\data_transformation\infer\img\scan_00002.png"
        with open(img_path, "rb") as f:
            self.client.post("/predict", files={"file": f})

    @task(1)
    def health_check(self):
        self.client.get("/health")