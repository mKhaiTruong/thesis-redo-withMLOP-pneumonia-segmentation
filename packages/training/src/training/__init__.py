from dataclasses import dataclass
from pathlib import Path

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