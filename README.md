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
UNet++ EfficientNetB3 · SegFormer ResNet50 · ONNX INT8 · FastAPI · DVC · MLflow · Prometheus · Grafana

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

[![CI/CD]](https://github.com/mKhaiTruong/thesis-redo-withMLOP-pneumonia-segmentation/actions)

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-yellow)](https://bill123mk-pneumonia-segmentation.hf.space/)

[![DagHub](https://img.shields.io/badge/MLflow-DagHub-orange)](https://dagshub.com/minhkhai0402/thesis-redo-withMLOP-pneumonia-segmentation.mlflow/)

---

## Demo

**Live:** [huggingface.co/spaces/bill123mk/pneumonia_segmentation](https://bill123mk-pneumonia-segmentation.hf.space/)

Upload a COVID-19 CT scan → get back a JET colormap segmentation mask highlighting infected lung regions.

<img width="2454" height="1156" alt="Image" src="https://github.com/user-attachments/assets/b93f9682-fe7d-48fa-8aec-90f9b29738e3" />

---

## Pipeline Overview

```
Stage 1   Data Ingestion        Download COVID-19 CT dataset
Stage 2   Data Transformation   NIfTI → PNG, crop lung ROI, JET colormap, train/valid/infer split
Stage 3   Data Drift Detection  Wasserstein distance vs baseline, per-request drift score in response headers
Stage 4   Prepare Base Model    Validate architecture config, generate model summary
Stage 5   Training              Dice + Focal loss, early stopping, MLflow logging → DagHub
Stage 6   ONNX Export           PyTorch → ONNX → INT8 quantization
Stage 7   Evaluation            ONNX Runtime inference on infer set, save metrics.json
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-gray)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-green)
![DVC](https://img.shields.io/badge/DVC-pipeline-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue)
![Docker](https://img.shields.io/badge/Docker-containerized-blue)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-orange)
![Grafana](https://img.shields.io/badge/Grafana-dashboard-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-black)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow)
![Data Drift](https://img.shields.io/badge/Data_Drift-Wasserstein-teal)

**Core:** Python 3.12 · PyTorch · segmentation-models-pytorch · ONNX Runtime

**Pipeline:** DVC · MLflow · DagHub

**Serving:** FastAPI · ONNX Runtime · Docker · HuggingFace Spaces

**Monitoring:** Prometheus · Grafana · prometheus-fastapi-instrumentator

**CI/CD:** GitHub Actions → HuggingFace Spaces

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

### 2. Run pipeline

```bash
activate.bat
dvc repro
```

To switch models, edit `params.yaml`:

```yaml
prepare_base_model_params:
model_name: "unetpp" # unet | unetpp | deeplabv3p | manet | segformer
encoder: "efficientnet-b4" # resnet50 | mobilenet_v2 | resnet34
```

Then `dvc repro` — outputs go to `artifacts/training/<model_name>_<encoder>/`.

### 3. Serve locally

```bash
activate.bat
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [localhost:8000](http://localhost:8000) for the inference UI.

### 4. Monitoring (local)

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

- Prometheus: [localhost:9090](http://localhost:9090)
- Grafana: [localhost:3000](http://localhost:3000)

**🔍 Monitoring Stack**

| Tool           | Role                                            |
| -------------- | ----------------------------------------------- |
| **FastAPI**    | Serves prediction endpoint & exposes `/metrics` |
| **Prometheus** | Scrapes metrics every 15s                       |
| **Grafana**    | Visualizes request rate, latency, error rate    |

### FastAPI — Swagger UI

<img width="2342" height="1286" alt="Image" src="https://github.com/user-attachments/assets/c3cdda2d-a714-4b61-9509-a59f1c86671a" />

### Prometheus — Target Scraping

<img width="2551" height="910" alt="Image" src="https://github.com/user-attachments/assets/755b9ff4-dc6a-4ac6-bc17-73d6b18d4268" />

### Grafana — Live Dashboard

<img width="2553" height="1320" alt="Image" src="https://github.com/user-attachments/assets/b9fdbfc2-ebc7-4a40-b25c-80693074aa15" />

<img width="2517" height="1180" alt="Image" src="https://github.com/user-attachments/assets/1634da7f-d35b-498b-8dc0-2d163866b67d" />
```
---

## Experiment Tracking

All runs logged to DagHub via MLflow:
[https://dagshub.com/minhkhai0402/thesis-redo-withMLOP-pneumonia-segmentation/experiments](https://dagshub.com/minhkhai0402/thesis-redo-withMLOP-pneumonia-segmentation/experiments)

Each run logs: `model_name`, `encoder`, `lr`, `batch_size`, `epochs`, `IOU`, `loss`.

<img width="2557" height="398" alt="Image" src="https://github.com/user-attachments/assets/25c3278f-5420-4a20-9c09-cf2bdfdf1c1e" />

deploy:
      resources:
        limits:
          memory: 2g
          cpus: '1.5'
<img width="2541" height="462" alt="Image" src="https://github.com/user-attachments/assets/e843177d-6424-4134-8eea-6211ba5e6d7f" />

---

## CI/CD

Push to `main` triggers:

```
pytest tests/  >>  deploy to HuggingFace Space
```

Weights and ONNX models are uploaded manually via `upload_to_hf.py` to [HF Hub](https://huggingface.co/bill123mk/pneumonia-seg-weights).

