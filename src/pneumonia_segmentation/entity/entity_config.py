from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Stage 1: Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:    Path
    source_type: str            
    source:      str     
    name:        str

# Stage 2: Data Transformation  
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir:       Path
    data_dirs:      tuple
    out_train_dir:  Path
    out_valid_dir:  Path
    out_infer_dir:  Path
    params_image_size: int
    params_skip_background_ratio: float
    params_slice_interval: int
    params_valid_size: float
    params_infer_size: float

# Stage 3: Data Drift
@dataclass(frozen=True)
class DataDriftMetricsConfig:
    drift_threshold:    float
    n_bins:             int
    max_samples:        Optional[int]
    seed:               int
    model_name:         str

@dataclass(frozen=True)
class DataDriftConfig:
    root_dir: Path
    origin_data_source: Path
    baseline_dir: Path
    metric: DataDriftMetricsConfig

# Stage 4: Prepare Base Model
@dataclass(frozen=True)
class ModelArchitecture:
    model_architecture: str
    library: str
    model_name: str
    encoder: str
    encoder_weights: str
    classes: int
    activation: str

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    modelArchitecture: ModelArchitecture

# Stage 5: Training
@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    encoder:    str
    encoder_weights:    str
    checkpoint_dir:     Path
    latest_model_dir:   Path
    best_model_dir:     Path
    run_info_dir:       Path

@dataclass(frozen=True)
class DataConfig:
    train_data_dir: Path
    valid_data_dir: Path

@dataclass(frozen=True)
class OptimizerConfig:
    lr:             float
    decay:          float
    lr_scheduler:   str

@dataclass(frozen=True)
class MetricConfig:
    metric_mode:    str
    loss_function:  str
    alpha:          float
    gamma:          float

@dataclass(frozen=True)
class TrainParamsConfig:
    batch_size:     int
    patience:       int
    start_epoch:    int
    epochs:         int
    workers:        int
    seed:           int
    image_size:     int
    is_augmentation: bool

@dataclass(frozen=True)
class TrainingConfig:
    root_dir:   Path
    model:      ModelConfig
    data:       DataConfig
    metric:     MetricConfig
    optimizer:  OptimizerConfig
    train:      TrainParamsConfig

# Stage 6: ONNX
@dataclass(frozen=True)
class TrainedModelConfig:
    checkpoint_dir  : Path
    latest_model_dir: Path
    best_model_dir  : Path
    run_info_dir    : Path

@dataclass(frozen=True)
class OnnxConfig:
    root_dir:               Path
    trained_model:          TrainedModelConfig
    onnx_model_dir:         Path
    onnx_int8_model_dir:    Path
    image_size:             int
 
# Stage 7: TensorRT
@dataclass(frozen=True)
class OnnxModelConfig:
    onnx_dir:       Path
    onnx_int8_dir:  Path
    
@dataclass(frozen=True)
class TensorRTConfig:
    root_dir: Path
    out_dir:  Path
    image_size: int
    onnx: OnnxModelConfig
    

# Stage 8: Evaluation
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
    onnx: OnnxModelConfig
    engine: TensorRTConfig
    data: EvalDataConfig
    eval: EvaluationParamsConfig