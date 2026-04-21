import kaggle

def trigger_training():
    kaggle.api.authenticate()
    kaggle.api.kernels_push("path/to/training/notebook")
    
    while True:
        status = kaggle.api.kernel_status("username/notebook-name")
        if status["status"] == "complete":
            break
        time.sleep(30)