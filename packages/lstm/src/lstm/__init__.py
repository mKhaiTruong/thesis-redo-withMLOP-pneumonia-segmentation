from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class LSTM_Params_Config:
    input_size:  int
    hidden_size: int
    num_layers:  int
    input_steps: int
    output_steps: int
    epochs:     int
    batch_size: int
    lr:         float

@dataclass(frozen=True)
class LSTM_Predictor_Config:
    root_dir:       Path
    model_dir:      Path
    lstm_params:    LSTM_Params_Config