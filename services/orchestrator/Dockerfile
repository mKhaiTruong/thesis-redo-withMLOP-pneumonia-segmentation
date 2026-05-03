FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/orchestrator ./packages/orchestrator
RUN pip install -e ./packages/orchestrator

COPY services/orchestrator ./services/orchestrator
COPY config ./config
COPY params.yaml .

ENV PREFECT_API_URL=http://prefect:4200/api

CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "7860"]