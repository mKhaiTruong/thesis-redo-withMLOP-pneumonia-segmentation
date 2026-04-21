import sys
from torch.utils.data import DataLoader

from core.exception import CustomException
from evaluation.utils.custom_dataset import CustomDataset
from evaluation import EvaluationConfig

def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__    # type: ignore[name-defined]
        return shell == 'ZMQInteractiveShell'  # Jupyter
    except NameError:
        return False  # standard Python
    
def get_eval_dataloader(config: EvaluationConfig):
    try:
        infer_ds = CustomDataset(
            images=sorted((config.data.infer_data_dir / "img").glob("*.png")),
            is_train=config.eval.is_augmentation,
            image_size=config.eval.image_size,
            masks=sorted((config.data.infer_data_dir / "msk").glob("*.png")),
        )
    except Exception as e:
        raise CustomException(e, sys)
    
    num_workers = 0 if _is_notebook() else config.eval.workers
    pin_memory  = not _is_notebook()
    
    infer_loader = DataLoader(
        infer_ds, batch_size=config.eval.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, 
        persistent_workers=num_workers > 0, 
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return infer_loader