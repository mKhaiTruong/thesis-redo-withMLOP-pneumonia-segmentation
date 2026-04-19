FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/lstm ./packages/lstm
RUN pip install -e ./packages/lstm

COPY services/lstm ./services/lstm
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.lstm.main:app", "--host", "0.0.0.0", "--port", "7860"]