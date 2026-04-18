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
    root_dir: Path
    origin_data_source: Path
    baseline_dir: Path
    metric: DataDriftMetricsConfig