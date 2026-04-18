FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY packages/core ./packages/core
RUN pip install -e ./packages/core

COPY packages/onnx_export ./packages/onnx_export
RUN pip install -e ./packages/onnx_export

COPY services/onnx_export ./services/onnx_export
COPY config ./config
COPY params.yaml .

CMD ["uvicorn", "services.onnx_export.main:app", "--host", "0.0.0.0", "--port", "7860"]