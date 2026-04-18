FROM python:3.12-slim

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/prepare_base_model ./packages/prepare_base_model
RUN pip install -e ./packages/prepare_base_model

COPY services/prepare_base_model ./services/prepare_base_model
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.prepare_base_model.main:app", "--host", "0.0.0.0", "--port", "7860"]