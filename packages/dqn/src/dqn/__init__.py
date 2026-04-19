from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Experience:
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       bool

@dataclass(frozen=True)
class Simulation_Config:
    cpu_warning:    int
    cpu_critical:   int
    ram_warning:    int
    ram_critical:   int
    latency_warning:  float
    latency_critical: float
    drift_warning:  float
    drift_critical: float

@dataclass(frozen=True)
class DuelingDQN_Params_Config:
    state_size:  int
    action_size: int
    hidden_size: int
    
@dataclass(frozen=True)
class DQN_Params_Config:
    lr:         float
    gamma:      float
    epsilon:    float
    eps_min:    float
    eps_decay:  float
    batch_size: int
    target_update_freq: int

@dataclass(frozen=True)
class DQN_Planner_Config:
    root_dir:   Path
    model_dir:  Path
    duel_dqn_params:    DuelingDQN_Params_Config
    dqn_params:         DQN_Params_Config