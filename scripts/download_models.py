import os
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="bill123mk/pneumonia-seg-weights",
    filename="resnet50.onnx",
    repo_type="model",
    token=os.environ.get("HF_TOKEN"),
    local_dir="."
)