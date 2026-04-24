FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/data_drift ./packages/data_drift
RUN pip install -e ./packages/data_drift

COPY packages/inference ./packages/inference
RUN pip install -e ./packages/inference

COPY ./app ./app
COPY ./config ./config
COPY params.yaml .
COPY prometheus.yml .
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY resnet50.onnx .
ENV PYTHONPATH=/app:/app/src

EXPOSE 7860

# CMD ["python3", "app.py"]
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]