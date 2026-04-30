from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ClaudeParams:
    model:                str
    max_tokens:           int
    confidence_threshold: float

@dataclass(frozen=True)
class ClaudeValidationConfig:
    root_dir:       Path
    history_path:   Path
    params:         ClaudeParams