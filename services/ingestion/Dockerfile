FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/ingestion ./packages/ingestion
RUN pip install -e ./packages/ingestion

COPY services/ingestion ./services/ingestion
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.ingestion.main:app", "--host", "0.0.0.0", "--port", "7860"]