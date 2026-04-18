from dataclasses import dataclass
from pathlib import Path

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