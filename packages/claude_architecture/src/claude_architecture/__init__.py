from dataclasses import dataclass

@dataclass(frozen=True)
class LLMParams:
    model:      str
    max_tokens: int

@dataclass(frozen=True)
class ClaudeArchitectureConfig:
    prometheus_url: str
    params:         LLMParams