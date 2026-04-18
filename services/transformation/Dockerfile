FROM python:3.12-slim

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/transformation ./packages/transformation
RUN pip install -e ./packages/transformation

COPY services/transformation ./services/transformation
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.transformation.main:app", "--host", "0.0.0.0", "--port", "7860"]