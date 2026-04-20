from abc import ABC, abstractmethod

class BaseAIManagerComponent(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass