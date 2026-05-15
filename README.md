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

![k3d](https://img.shields.io/badge/Kubernetes-k3d-326CE5?logo=kubernetes)
![Traefik](https://img.shields.io/badge/Ingress-Traefik-EF3B24?logo=traefikproxy)
![Claude](https://img.shields.io/badge/AI_Architect-Claude_Haiku-8A2BE2)

---

## Demo

**Live:** [huggingface.co/spaces/bill123mk/pneumonia_segmentation](https://bill123mk-pneumonia-segmentation.hf.space/)

Upload a COVID-19 CT scan → get back a segmentation mask highlighting infected lung regions (JET/BONE colormap, configurable via `params.yaml`).

With lung input:
<div align="center">
  <img width="75%" alt="Image" src="https://github.com/user-attachments/assets/cea985a9-40c1-4bdb-b96e-0c2545370e00" />
</div>

With anomaly input:
<div align="center">
  <img width="75%" alt="Image" src="https://github.com/user-attachments/assets/b26534d9-190b-4ef3-8a44-43ef4e50ded5" />
</div>

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

<div align="center">
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/1f1a46e7-df4d-4e81-82be-d50fb462d3f0" />
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/a8dd110a-c3a3-4478-be8e-b882780e6e6e" />
</div>

<div align="center">
  <img width="50%" alt="Image" src="https://github.com/user-attachments/assets/e012e0cc-7eda-4102-ac4d-fae69d06f660" />
</div>


---

## Architecture

```
Local                 Kaggle (training)                  Local / Cloud (always-on)
─────                 ─────────────────                  ──────────────────────────
service flow:         train                               app          (FastAPI inference)
ingestion ->          -> onnx export            ->        ai_manager   (MAPE-K loop)
transformation ->     -> eval                             orchestrator (Prefect pipeline)
data_drift            -> upload HF Model                  lstm         (state prediction)
                                                          dqn          (action planning) (legacy)
                                                          claude-arch  (autonomous decisions)
data drift result ->                                      prometheus   (metrics scraping)
  Huggingface Models                                      grafana      (dashboard)
                                                          prefect      (workflow UI)
transformation result
  -> Kaggle Dataset
```
---

### MAPE-K Autonomous Loop (every 5 min)

```
Monitor  -> query Prometheus: ram, latency, drift_score, requests
Analyze  -> LSTM: predict next 5 minutes from 30-step time series
output: predicted_{ram, latency, drift, requests}
Plan     -> Claude Architecture: reason over current + predicted state
proactive decisions — act before thresholds are breached
Execute  -> cooldown check (300s) → POST orchestrator/execute/{action}

Actions:
  spawn     → scale_out_service     (High latency)     
  swap      → swap_model_version    (drift > 0.6 OR predicted to breach)
  rollback  → rollback              (latency spiked >2x after recent action)
  none      → all metrics within thresholds

Claude Architecture maintains decision history (last 10 decisions) and reasons
with full context: current metrics + LSTM predictions + past actions.
```

#### Legacy: DQN Action Planning *(experimental, not in production)*

```
[LEGACY] Plan → DQN: select action from 24-feature state vector
trained on synthetic data, replaced by Claude Architecture
reasoning due to reward signal limitations in production

Actions:
  trigger_retraining      → Prefect retrain_flow (3-node pipeline)
  switch_to_lighter_model → POST app/switch-model
  scale_up_service        → run full ml_pipeline
  restart_service         → POST app/reload-model
```
---

## Kubernetes Deployment (k3d)

Services run on a local k3d cluster with production-grade configuration:

- **Ingress:** Traefik with strip-prefix middleware, accessible via `pneumonia.local`
- **Storage:** PersistentVolumes for model artifacts, Grafana state, decision history
- **Secrets:** Injected via `kubectl create secret --from-env-file`, never hardcoded
- **RBAC:** ServiceAccount + ClusterRole for orchestrator pod-management permissions
- **Observability:** Prometheus scraping all services, Grafana dashboards provisioned via API


### Prefect Retraining Pipeline

```
Node 1 [Auto]:   ingestion → transformation → data_drift
Node 2 [Pause]:  wait for dev to train on Kaggle → Resume when done
Node 3 [Auto]:   reload model from HF
```

<div align="center">
  <img width="32%" alt="Image" src="https://github.com/user-attachments/assets/e547c26d-d031-4b24-bea5-6804da9fe5f6" />
  <img width="32%" alt="Image" src="https://github.com/user-attachments/assets/24f613ad-15df-4888-b144-b5133562b7ba" />
  <img width="32%" alt="Image" src="https://github.com/user-attachments/assets/4ee96086-4f9b-4a64-a7f4-b5dbdc841948" />
</div>


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
![Anthropic](https://img.shields.io/badge/Anthropic-Claude_Haiku-8A2BE2)
![k3d](https://img.shields.io/badge/k3d-Kubernetes-326CE5?logo=kubernetes)

**Core:** Python 3.12 · PyTorch · segmentation-models-pytorch · ONNX Runtime

**Pipeline:** DVC · MLflow · DagsHub · Prefect 3 · kagglehub

**Serving:** FastAPI · ONNX Runtime · Docker · HuggingFace Spaces

**Monitoring:** Prometheus · Grafana · prometheus-fastapi-instrumentator

**AI Loop:** LSTM (metric forecasting) · Claude Haiku (autonomous architectural reasoning) · MAPE-K

**Infrastructure:** minikube · k3d · kubedge · Traefik · RBAC · PersistentVolumes

**CI/CD:** GitHub Actions → HuggingFace Spaces (auto-deploy on push to main)

---

## Quickstart

> ⚠️ This project is a research/thesis platform — full local setup requires
> k3d, Docker, Kaggle API credentials, HuggingFace token, and DagsHub access.
> The sections below outline the key entry points.

### ML Pipeline (DVC)
```bash
cp .env.example .env  # fill in KAGGLE, DAGSHUB, HUGGINGFACE tokens
dvc repro
```

### Inference (HuggingFace Space)
Live demo — no setup required:
[bill123mk-pneumonia-segmentation.hf.space](https://bill123mk-pneumonia-segmentation.hf.space/)

### Microservices (k3d)
```bash
k3d cluster create pneumonia -p "80:80@loadbalancer" -p "443:443@loadbalancer"
kubectl apply -f k8s/
curl -X POST http://pneumonia.local/lstm/train  # train LSTM on first run
```

---

## Monitoring

<div align="center">
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/755b9ff4-dc6a-4ac6-bc17-73d6b18d4268" />
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/1634da7f-d35b-498b-8dc0-2d163866b67d" />
</div>

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

## Monitoring With Custom Dashboard and Grafana Pre-defined Visualizations

By running locustfile, the project mimics requests and users, though it is not accurate. There are 2 dashboards: Grafana and my own. Grafana covers all the visualizations, querying Prometheus's metrics. My custom app shows buttons that run locust file at different latency, requests/s, or drift score.

<div align="center">
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/4c98c737-b386-4164-88f9-6c4124cb4ffa" />
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/49803382-579d-4c10-af66-ed6674a84874" />
</div>

---

<div align="center">
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/d8f04e57-5448-419f-ae60-079d0364bac8" />
  <img width="49%" alt="Image" src="https://github.com/user-attachments/assets/c7e94258-4f01-44cf-8213-43e316534db4" />
</div>

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

- INT8 quantization slower than FP32 for UNet++ (SegFormer ops not efficiently quantized)
- Kaggle GPU training requires manual trigger (GPU selection not supported via API)
- MAPE-K cooldown: 300s between same action to prevent spam
- Edge Deployment is complete bust: Able to connect to Pi 5 through SSH, but Segformer Mit b5 + KubeEdge is too heavy for Pi

---