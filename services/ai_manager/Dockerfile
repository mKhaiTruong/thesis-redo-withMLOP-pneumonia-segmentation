FROM python:3.12-slim

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/ai_manager ./packages/ai_manager
RUN pip install -e ./packages/ai_manager

COPY services/ai_manager ./services/ai_manager
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.ai_manager.main:app", "--host", "0.0.0.0", "--port", "7860"]