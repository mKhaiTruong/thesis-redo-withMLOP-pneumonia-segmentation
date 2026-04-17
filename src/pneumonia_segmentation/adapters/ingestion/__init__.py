from abc import ABC, abstractmethod

class BaseDataIngestionAdapter(ABC):
    @abstractmethod
    def fetch(self, dst: str) -> None:
        pass