from dataclasses import dataclass

@dataclass(frozen=True)
class ClaudeParams:
    model:                str
    max_tokens:           int
    confidence_threshold: float

@dataclass(frozen=True)
class ClaudeValidationConfig:
    params: ClaudeParams