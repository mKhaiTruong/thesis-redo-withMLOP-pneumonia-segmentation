from abc import ABC, abstractmethod
import torch.nn as nn

class BuildModelStrategy(ABC):
    @abstractmethod
    def build_model(self, config) -> nn.Module:
        pass