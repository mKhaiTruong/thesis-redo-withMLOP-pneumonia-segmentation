from abc import ABC, abstractmethod
from typing import Generator, Tuple
import numpy as np

class BaseDataIngestionAdapter(ABC):
    @abstractmethod
    def fetch(self, dst: str) -> None:
        pass

class BaseDataSourceAdapter(ABC):
    @abstractmethod
    def get_data_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass
    
    @abstractmethod
    def get_total_count(self) -> int:
        pass