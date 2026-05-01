from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LLMParams:
    model:      str
    max_tokens: int

@dataclass(frozen=True)
class ClaudeArchitectureConfig:
    root_dir:           Path
    history_path:       Path
    prometheus_url:     str
    orchestrator_url:   str
    params:         LLMParams