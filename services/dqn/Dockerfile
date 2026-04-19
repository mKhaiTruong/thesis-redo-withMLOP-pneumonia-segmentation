FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/dqn ./packages/dqn
RUN pip install -e ./packages/dqn

COPY services/dqn ./services/dqn
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.dqn.main:app", "--host", "0.0.0.0", "--port", "7860"]