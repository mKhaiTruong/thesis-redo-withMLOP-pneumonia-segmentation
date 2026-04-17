from dataclasses import dataclass
from pathlib import Path

# Stage 1: Data Ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:    Path
    source_type: str            
    source:      str     
    name:        str