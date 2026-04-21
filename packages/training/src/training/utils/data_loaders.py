import sys
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

from core.exception import CustomException
from training.utils.custom_dataset import CustomDataset
from training import TrainingConfig

def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__    # type: ignore[name-defined]
        return shell == 'ZMQInteractiveShell'  # Jupyter
    except NameError:
        return False  # standard Python
    
def _get_sampler(dataset: CustomDataset):
    weights = []
    for msk_path in dataset.msk_paths:
        import cv2, numpy as np
        msk = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        weights.append(1.0 if msk.sum() > 0 else 0.2)
    return WeightedRandomSampler(weights, len(weights))

def get_dataloaders(config: TrainingConfig) -> dict:
    try:
        train_ds = CustomDataset(
            images=sorted((config.data.train_data_dir / "img").glob("*.png")),
            is_train=config.train.is_augmentation,
            image_size=config.train.image_size,
            masks=sorted((config.data.train_data_dir / "msk").glob("*.png"))
        )
        valid_ds = CustomDataset(
            images=sorted((config.data.valid_data_dir / "img").glob("*.png")),
            image_size=config.train.image_size,
            masks=sorted((config.data.valid_data_dir / "msk").glob("*.png")),
        )
    except Exception as e:
        raise CustomException(e, sys)
    
    num_workers = 0 if _is_notebook() else config.train.workers
    pin_memory  = not _is_notebook()
    
    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size, sampler=_get_sampler(train_ds),
        num_workers=num_workers, pin_memory=pin_memory, 
        persistent_workers=num_workers > 0, 
        prefetch_factor=2 if num_workers > 0 else None
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.train.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, 
        persistent_workers=num_workers > 0, 
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return {
        "train": train_loader,
        "valid": valid_loader
    }