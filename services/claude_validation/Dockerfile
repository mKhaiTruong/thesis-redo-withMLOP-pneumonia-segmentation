FROM python:3.12-slim

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/claude_validation ./packages/claude_validation
RUN pip install -e ./packages/claude_validation

COPY services/claude_validation ./services/claude_validation
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.claude_validation.main:app", "--host", "0.0.0.0", "--port", "7860"]