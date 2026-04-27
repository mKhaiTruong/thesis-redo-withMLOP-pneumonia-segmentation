---
title: Pneumonia Segmentation
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Pneumonia Segmentation — MLOps Pipeline

End-to-end MLOps pipeline for COVID-19 CT scan lung infection segmentation.  
Segmentation Model Pytorch · ONNX INT8 · FastAPI · DVC · MLflow · Prefect · Prometheus · Grafana · MAPE-K

[![CI/CD](https://github.com/mKhaiTruong/thesis-redo-withMLOP-pneumonia-segmentation/actions/workflows/workflow.yaml/badge.svg)](https://github.com/mKhaiTruong/thesis-redo-withMLOP-pneumonia-segmentation/actions)

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://bill123mk-pneumonia-segmentation.hf.space/)

[![DagsHub](https://img.shields.io/badge/MLflow-DagsHub-orange)](https://dagshub.com/minhkhai0402/thesis-redo-withMLOP-pneumonia-segmentation.mlflow/)

---

## Demo

**Live:** [huggingface.co/spaces/bill123mk/pneumonia_segmentation](https://bill123mk-pneumonia-segmentation.hf.space/)

Upload a COVID-19 CT scan → get back a segmentation mask highlighting infected lung regions (JET/BONE colormap, configurable via `params.yaml`).

With lung input:
<img width="2470" height="1172" alt="Image" src="https://github.com/user-attachments/assets/cea985a9-40c1-4bdb-b96e-0c2545370e00" />

With anomaly input:
<img width="2452" height="1154" alt="Image" src="https://github.com/user-attachments/assets/b26534d9-190b-4ef3-8a44-43ef4e50ded5" />

---

## Pipeline Overview

```
Stage 1   Data Ingestion        Download COVID-19 CT datasets from Kaggle (2 sources)
Stage 2   Data Transformation   NIfTI/PNG → crop lung ROI → colormap → train/valid/infer split
                                → push transformed data to Kaggle Dataset via kagglehub
Stage 3   Data Drift Detection  ResNet50 feature extraction → L2 norm vs baseline
                                → push baseline_distribution.npy to HF Model repo
Stage 4   Prepare Base Model    Validate architecture config, generate model summary
Stage 5   Training              UNet++ EfficientNet-B5 · Dice + Focal loss · early stopping
                                → resume from checkpoint · MLflow logging → DagsHub
                                → train on Kaggle P100 GPU (30h/week free tier)
Stage 6   ONNX Export           PyTorch → ONNX FP32 → INT8 dynamic quantization
Stage 7   TensorRT              ONNX -> TensorRT Engine (No edge GPU -> discarded)
Stage 8   Evaluation            ONNX Runtime benchmark: FP32/INT8 × CPU/CUDA
```

---

## Evaluation Results

Latest model: **UNet++ EfficientNet-B5** (90 epochs, IoU best: 0.7361 checkpoint → continued training)

| Model          | IoU    | Loss   | ms/batch |
| -------------- | ------ | ------ | -------- |
| onnx_fp32_cpu  | 0.8187 | 0.1006 | 4808.95  |
| onnx_int8_cpu  | 0.7113 | 0.1697 | 10999.28 |
| onnx_fp32_cuda | 0.8187 | 0.1006 | 673.89   |
| onnx_int8_cuda | 0.7102 | 0.1701 | 10503.79 |

> INT8 is slower than FP32 due to UNet++ MatMul layers not being efficiently quantized. FP32 is used for production inference.

<img width="2504" height="1002" alt="Image" src="https://github.com/user-attachments/assets/1f1a46e7-df4d-4e81-82be-d50fb462d3f0" />

<img width="944" height="700" alt="Image" src="https://github.com/user-attachments/assets/e012e0cc-7eda-4102-ac4d-fae69d06f660" />

<img width="2486" height="1186" alt="Image" src="https://github.com/user-attachments/assets/a8dd110a-c3a3-4478-be8e-b882780e6e6e" />

---

## Architecture

```
Local                 Kaggle (training)                  Local / Cloud (always-on)
─────                 ─────────────────                  ──────────────────────────
service flow:         train                               app          (FastAPI inference)
ingestion ->          -> onnx export            ->        ai_manager   (MAPE-K loop)
transformation ->     -> eval                             orchestrator (Prefect pipeline)
data_drift            -> upload HF Model                  lstm         (state prediction)
                                                          dqn          (action planning)
data drift result ->                                      prometheus   (metrics scraping)
  Huggingface Models                                      grafana      (dashboard)
                                                          prefect      (workflow UI)
transformation result
  -> Kaggle Dataset
```

---

### MAPE-K Autonomous Loop (every 5 min)

```
Monitor  → query Prometheus: cpu, ram, latency, drift_score
Analyze  → LSTM: predict next 5 steps from 30-step time series
Plan     → DQN: select action from 24-feature state vector
Execute  → cooldown check (300s) → POST orchestrator/execute/{action}

Actions:
  trigger_retraining      → Prefect retrain_flow (3-node pipeline)
  switch_to_lighter_model → POST app/switch-model
  scale_up_service        → run full ml_pipeline
  restart_service         → POST app/reload-model
```

---

### Prefect Retraining Pipeline

```
Node 1 [Auto]:   ingestion → transformation → data_drift
Node 2 [Pause]:  wait for dev to train on Kaggle → Resume when done
Node 3 [Auto]:   reload model from HF
```

<img width="2543" height="1171" alt="Image" src="https://github.com/user-attachments/assets/e547c26d-d031-4b24-bea5-6804da9fe5f6" />

<img width="2100" height="1120" alt="Image" src="https://github.com/user-attachments/assets/24f613ad-15df-4888-b144-b5133562b7ba" />

<img width="2248" height="1168" alt="Image" src="https://github.com/user-attachments/assets/4ee96086-4f9b-4a64-a7f4-b5dbdc841948" />

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-green)
![DVC](https://img.shields.io/badge/DVC-pipeline-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Prefect](https://img.shields.io/badge/Prefect-3-darkblue)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-dashboard-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

**Core:** Python 3.12 · PyTorch · segmentation-models-pytorch · ONNX Runtime

**Pipeline:** DVC · MLflow · DagsHub · Prefect 3 · kagglehub

**Serving:** FastAPI · ONNX Runtime · Docker · HuggingFace Spaces

**Monitoring:** Prometheus · Grafana · prometheus-fastapi-instrumentator

**AI Loop:** LSTM (state prediction) · DQN (action planning) · MAPE-K

**CI/CD:** GitHub Actions → HuggingFace Spaces (auto-deploy on push to main)

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/mKhaiTruong/thesis-redo-withMLOP-pneumonia-segmentation
cd thesis-redo-withMLOP-pneumonia-segmentation
pip install pip-tools
pip-compile requirements-dev.in
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Setup environment

```bash
cp .env.example .env
# Fill in: KAGGLE_USERNAME, KAGGLE_API_TOKEN, DAGSHUB_TOKEN,
#          HUGGING_FACE_TOKEN, MLFLOW_TRACKING_URI
```

### 3. Run pipeline (DVC)

```bash
dvc repro
```

To switch models, edit `params.yaml`:

```yaml
prepare_base_model_params:
  model_name: "unetpp" # unet | unetpp | deeplabv3 | manet | segformer
  encoder: "efficientnet-b5" # efficientnet-b5 | mit_b2 | resnet50 | ...
```

Then `dvc repro` — outputs go to `artifacts/training/<model_name>_<encoder>/`.

### 4. Run microservices (Docker)

```bash
# Start data pipeline services
docker compose up -d prefect ingestion transformation data_drift orchestrator

# Start AI loop
docker compose up -d lstm dqn ai_manager

# Start monitoring
docker compose up -d prometheus grafana
```

Services:
| Service | Port | Role |
|----------------|------|-------------------------------|
| ingestion | 7861 | Download Kaggle datasets |
| transformation | 7862 | Process NIfTI/PNG → training data |
| data_drift | 7863 | Compute baseline & drift score |
| orchestrator | 7867 | Prefect pipeline coordinator |
| lstm | 7868 | State prediction |
| dqn | 7869 | Action planning |
| ai_manager | 7870 | MAPE-K loop (every 5 min) |
| prefect | 4200 | Workflow UI |
| prometheus | 9090 | Metrics scraping |
| grafana | 3000 | Dashboard |

### 5. Trigger retraining pipeline

```bash
curl -X POST http://localhost:7867/execute/trigger_retraining
```

Then go to [localhost:4200](http://localhost:4200) → watch Node 1 complete → train on Kaggle → Resume in Prefect UI.

### 6. Serve locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open [localhost:7860](http://localhost:7860) for the inference UI.

---

## Monitoring

<img width="2551" height="910" alt="Image" src="https://github.com/user-attachments/assets/755b9ff4-dc6a-4ac6-bc17-73d6b18d4268" />

<img width="2517" height="1180" alt="Image" src="https://github.com/user-attachments/assets/1634da7f-d35b-498b-8dc0-2d163866b67d" />

| Tool       | Role                                          |
| ---------- | --------------------------------------------- |
| FastAPI    | Serves `/predict` · exposes `/metrics`        |
| Prometheus | Scrapes metrics every 15s                     |
| Grafana    | Visualizes latency, drift score, request rate |

Drift score is returned in every prediction response header:

```
X-Drift-Score: 12.3456
X-Drift-Status: ok | warn | fail
X-Drift-Detected: 0 | 1
X-Inference-Ms: 145.2
```

---

## Experiment Tracking

All runs logged to DagsHub via MLflow:  
[dagshub.com/minhkhai0402/.../experiments](https://dagshub.com/minhkhai0402/thesis-redo-withMLOP-pneumonia-segmentation.mlflow/)

Each run logs: `model_name`, `encoder`, `lr`, `batch_size`, `epochs`, `IoU`, `loss`.

## Model Weights

Hosted on HuggingFace Model Hub: [bill123mk/pneumonia-seg-weights](https://huggingface.co/bill123mk/pneumonia-seg-weights)

```
pneumonia-seg-weights/
├── best_model_int8.onnx          ← production inference (HF Space)
├── best_model_fp32.onnx
├── baseline_distribution.npy     ← drift detection baseline
└── unetpp_efficientnet-b5/
    ├── best_model.pth
    ├── model.pth
    └── run_info.json
```

Weights are automatically uploaded after each Kaggle training run via `scripts/upload_to_hf.py`.

---

## CI/CD

Push to `main` → GitHub Actions:

```
pytest tests/ → deploy to HuggingFace Space
```

HF Space automatically downloads model weights from HF Model Hub on startup — no large files in git repo.

## Known Limitations

- INT8 quantization slower than FP32 for UNet++ (MatMul ops not efficiently quantized)
- Kaggle GPU training requires manual trigger (GPU selection not supported via API)
- MAPE-K cooldown: 300s between same action to prevent spam
- Free HF Space shared RAM — system RAM may appear high due to other tenants
