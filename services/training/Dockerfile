FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/training ./packages/training
RUN pip install -e ./packages/training

COPY services/training ./services/training
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.training.main:app", "--host", "0.0.0.0", "--port", "7860"]