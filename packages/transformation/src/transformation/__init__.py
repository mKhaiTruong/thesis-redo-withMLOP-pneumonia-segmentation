from dataclasses import dataclass
from pathlib import Path

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