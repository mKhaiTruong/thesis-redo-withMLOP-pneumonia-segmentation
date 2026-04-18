FROM python:3.12-slim

WORKDIR /app

COPY resnet50.onnx .

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/data_drift ./packages/data_drift
RUN pip install -e ./packages/data_drift

COPY services/data_drift ./services/data_drift
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.data_drift.main:app", "--host", "0.0.0.0", "--port", "7860"]