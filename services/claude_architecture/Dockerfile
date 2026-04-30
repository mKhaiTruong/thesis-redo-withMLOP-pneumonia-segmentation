FROM python:3.12-slim

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/claude_architecture ./packages/claude_architecture
RUN pip install -e ./packages/claude_architecture

COPY services/claude_architecture ./services/claude_architecture
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.claude_architecture.main:app", "--host", "0.0.0.0", "--port", "7860"]