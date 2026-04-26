import os

APP_URL         = "http://app:7860"
MICROSERVICES   = {
    "ingestion":      "http://ingestion:7860/run-ingestion",
    "transformation": "http://transformation:7860/run-transformation",
    "data_drift":     "http://data_drift:7860/run-data-drift",
}

HF_REPO_ID      = "bill123mk/pneumonia-seg-weights"
HF_STATUS_FILE  = "retrain_status.json"
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KERNEL   = os.getenv("KAGGLE_KERNEL_SLUG")
POLL_INTERVAL   = 60 * 2
POLL_TIMEOUT    = 3600 * 10