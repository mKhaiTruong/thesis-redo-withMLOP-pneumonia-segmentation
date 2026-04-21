from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    model_dir:       Path

@dataclass(frozen=True)
class OnnxModelConfig:
    onnx_dir:       Path
    onnx_int8_dir:  Path

@dataclass(frozen=True)
class EvalDataConfig:
    infer_data_dir: Path
    
@dataclass(frozen=True)
class EvaluationParamsConfig:
    batch_size: int
    workers:    int
    image_size: int
    is_augmentation: bool
    threshold:  float

@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    model:  ModelConfig
    onnx:   OnnxModelConfig
    data:   EvalDataConfig
    eval:   EvaluationParamsConfig