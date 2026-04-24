from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass(frozen=True)
class DataDriftMetricsConfig:
    drift_threshold:    float
    n_bins:             int
    max_samples:        Optional[int]
    seed:               int
    model_name:         str

@dataclass(frozen=True)
class DataDriftConfig:
    root_dir:           Path
    origin_data_source: Path
    baseline_dir:       Path
    metric:             DataDriftMetricsConfig

@dataclass(frozen=True)
class ModelConfig:
    model_name:      str    
    encoder:         str
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