from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class OnnxModelConfig:
    onnx_dir:       Path
    onnx_int8_dir:  Path
    
@dataclass(frozen=True)
class TensorRTEngineConfig:
    engine_dir:  Path

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
    onnx:   OnnxModelConfig
    engine: TensorRTEngineConfig
    data:   EvalDataConfig
    eval:   EvaluationParamsConfig