from abc import ABC, abstractmethod
from typing import Generator

import os
from core.logging import logger

class BaseDataTransformationAdapter(ABC):
    @property
    @abstractmethod
    def REQUIRED_SUBDIRS(self) -> dict: pass
    
    @property
    @abstractmethod
    def OPTIONAL_SUBDIRS(self) -> dict: pass
    
    def validate_and_init(self):
        self._validate_structure()
        self.align_datasets()
    
    def _validate_structure(self):
        missing = [
            f"  - {subdir}/  ({desc})"
            for subdir, desc in self.REQUIRED_SUBDIRS.items()
            if not os.path.isdir(os.path.join(self.data_dir, subdir))
        ]
        
        if missing:
            raise FileNotFoundError(
                f"{self.__class__.__name__} missing required folders:\n" + "\n".join(missing)
            )
            
        self.has_lung_mask = os.path.isdir(os.path.join(self.data_dir, "lung_mask"))
        if not self.has_lung_mask:
            logger.warning("lung_mask not found — will use full_slice instead of crop_to_lung_roi")
    
    @abstractmethod
    def align_datasets(self) -> None:   pass
    
    @abstractmethod
    def get_total_count(self) -> int:   pass
    
    @abstractmethod
    def get_data_generator(self) -> Generator: pass