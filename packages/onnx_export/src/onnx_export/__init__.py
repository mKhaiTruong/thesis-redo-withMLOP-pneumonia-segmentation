from dataclasses import dataclass
from pathlib import Path

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