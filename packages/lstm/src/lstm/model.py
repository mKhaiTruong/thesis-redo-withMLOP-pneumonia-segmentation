import torch
import torch.nn as nn

class LSTM_Predictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_steps=5):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, input_size * output_steps)
        
        self.output_steps = output_steps
        self.input_size   = input_size
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.fc(out[:, -1, :])
        out    = out.view(-1, self.output_steps, self.input_size)
        return out