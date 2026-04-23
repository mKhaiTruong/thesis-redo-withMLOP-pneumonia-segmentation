import os
import json
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

from core.logging import logger

load_dotenv()
# -- config --
REPO_ID    = "bill123mk/pneumonia-seg-weights"
TRAIN_ROOT = Path("artifacts/training")
ONNX_ROOT  = Path("artifacts/onnx")

# -- resolve slug from run_info.json --
login(token=os.getenv("HUGGING_FACE_TOKEN"))
api = HfApi()
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

for run_info_path in TRAIN_ROOT.glob("*/run_info.json"):
    slug     = run_info_path.parent.name
    files    = [
        (TRAIN_ROOT / slug / "best_model.pth",  f"{slug}/best_model.pth"),
        (TRAIN_ROOT / slug / "model.pth",        f"{slug}/model.pth"),
        (TRAIN_ROOT / slug / "run_info.json",    f"{slug}/run_info.json"),
        (ONNX_ROOT  / slug / "model.onnx",       f"{slug}/model.onnx"),
        (ONNX_ROOT  / slug / "model_int8.onnx",  f"{slug}/model_int8.onnx"),
    ]
    for local_path, repo_path in files:
        if not local_path.exists():
            logger.info(f"Skipping {local_path} - not found")
            continue
        api.upload_file(
            path_or_fileobj = str(local_path),
            path_in_repo    = repo_path,
            repo_id         = REPO_ID,
        )
        logger.info(f"Uploaded {local_path} -> {repo_path}")

# -- push chosen model flat --
from core.constants import * 
from core.utils import read_yaml
config = read_yaml(CONFIG_FILE_PATH)

BEST_SLUG = max(
    TRAIN_ROOT.glob("*/run_info.json"),
    key=lambda p: json.loads(p.read_text()).get("iou_score", 0)
).parent.name

best_onnx   = ONNX_ROOT / BEST_SLUG / "model.onnx"
best_onnx_int8 = ONNX_ROOT / BEST_SLUG / "model_int8.onnx"

if best_onnx.exists():
    api.upload_file(
        path_or_fileobj = str(best_onnx),
        path_in_repo    = "best_model.onnx",
        repo_id         = REPO_ID,
    )
    logger.info(f"Best model set to: {BEST_SLUG}")
else:
    logger.info(f"WARNING: {best_onnx} not found - best_model.onnx not updated")
    
if best_onnx_int8.exists():
    api.upload_file(
        path_or_fileobj = str(best_onnx_int8),
        path_in_repo    = "best_model_int8.onnx",
        repo_id         = REPO_ID,
    )
    logger.info(f"Best model set to: {BEST_SLUG}")
else:
    logger.info(f"WARNING: {best_onnx_int8} not found - best_model_int8.onnx not updated")

logger.info(f"Done! https://huggingface.co/{REPO_ID}")


# -- push drift baseline --
BASELINE_PATH = Path("artifacts/data_drift/baseline_distribution.npy")

if BASELINE_PATH.exists():
    api.upload_file(
        path_or_fileobj = str(BASELINE_PATH),
        path_in_repo    = "baseline_distribution.npy",
        repo_id         = REPO_ID,
    )
    logger.info("Baseline distribution uploaded.")
else:
    logger.info("WARNING: baseline_distribution.npy not found - skipping")
    
# -- push retrain status signal --
import json, tempfile, time
status_payload = json.dump({
    "status":    "complete",
    "slug":      BEST_SLUG,
    "timestamp": time.time(),
})

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write(status_payload)
    tmp_path = f.name

api.upload_file(
    path_or_fileobj = tmp_path,
    path_in_repo    = "retrain_status.json",
    repo_id         = REPO_ID,
)
logger.info("Retrain status signal pushed to HF.")