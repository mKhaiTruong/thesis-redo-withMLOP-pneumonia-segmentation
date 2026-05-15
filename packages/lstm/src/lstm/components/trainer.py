import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from core.logging import logger
from lstm import LSTM_Predictor_Config
from lstm.model import LSTM_Predictor
from lstm.utils.data.synthetic_generator import generate_synthetic_metrics
from lstm import METRIC_NAMES

class LSTM_Trainer:
    def __init__(self, config: LSTM_Predictor_Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._get_model()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lstm_params.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=10, factor=0.5, min_lr=1e-6)
        
        self.mean: torch.Tensor | None = None
        self.std:  torch.Tensor | None = None
        self.loader    = self._prepare_loader()     # WARNING: prepare_loader must be after mean and std
    
    def _get_model(self):
        return LSTM_Predictor(
            input_size   = self.config.lstm_params.input_size,
            hidden_size  = self.config.lstm_params.hidden_size,
            num_layers   = self.config.lstm_params.num_layers,
            output_steps = self.config.lstm_params.output_steps
        ).to(self.device)
    
    def _prepare_loader(self) -> DataLoader:
        df   = generate_synthetic_metrics()
        
        # Verify column order matches METRICS definition
        assert list(df.columns) == METRIC_NAMES, (
            f"synthetic_generator columns {list(df.columns)} "
            f"do not match METRIC_NAMES {METRIC_NAMES}. "
            f"Fix synthetic_generator.py or METRICS list."
        )
        
        data = torch.tensor(df.values, dtype=torch.float32)
        
        # Z-score normalization — params saved into checkpoint for inference
        self.mean = data.mean(dim=0)
        self.std  = data.std(dim=0)
        data = (data - self.mean) / (self.std + 1e-8)
        
        inp_steps = self.config.lstm_params.input_steps
        out_steps = self.config.lstm_params.output_steps
        
        X, y = [], []
        for i in range(len(data) - inp_steps - out_steps):
            X.append(data[i : i + inp_steps])
            y.append(data[i + inp_steps : i + inp_steps + out_steps])
        
        X = torch.stack(X)
        y = torch.stack(y)
        
        logger.info(
            f"Training data prepared | "
            f"samples={len(X)} | features={METRIC_NAMES} | "
            f"mean={self.mean.tolist()} | std={self.std.tolist()}"
        )
        
        return DataLoader(
            TensorDataset(X, y), 
            batch_size=self.config.lstm_params.batch_size, 
            shuffle=True
        )
    
    def train(self):
        epochs = self.config.lstm_params.epochs
        for epoch in range(epochs):
            avg_loss = self._train_one_epoch()
            self.scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} "
                    f"— Loss: {avg_loss:.6f} "
                    f"— LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        self._save_model()
    
    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in self.loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
                
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss   = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.loader)
    
    def _save_model(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mean":         self.mean,
            "std":          self.std,
            "metric_names": METRIC_NAMES,          # sanity check on load
            "input_steps":  self.config.lstm_params.input_steps,
            "output_steps": self.config.lstm_params.output_steps,
            "hidden_size":  self.config.lstm_params.hidden_size,
            "num_layers":   self.config.lstm_params.num_layers,
        }, self.config.model_dir)
        
        logger.info(f"LSTM model saved -> {self.config.model_dir}")