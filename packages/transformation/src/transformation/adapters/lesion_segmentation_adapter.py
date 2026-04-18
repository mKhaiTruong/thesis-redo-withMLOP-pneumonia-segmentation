import os, cv2, numpy as np
from tqdm import tqdm
from core.logging import logger
from transformation.adapters import BaseDataTransformationAdapter

class LesionSegmentationAdapter(BaseDataTransformationAdapter):
    REQUIRED_SUBDIRS = {
        "Dataset/Annotations":  "Infection segmentation masks",
        "Dataset/Images":       "CT scan volumes",
    }
    OPTIONAL_SUBDIRS = {
        "lung_mask": "Lung segmentation masks (optional)",
    }
    
    def __init__(self, data_dir: str):
        self.data_dir       = data_dir
        self.ct_dir         = os.path.join(data_dir, "Dataset", "Images")
        self.infection_dir  = os.path.join(data_dir, "Dataset", "Annotations")
        self.validate_and_init()
    
    def align_datasets(self) -> None:
        ct_names      = {os.path.splitext(f)[0] for f in os.listdir(self.ct_dir)        if f.endswith(".png")}
        infect_names  = {os.path.splitext(f)[0] for f in os.listdir(self.infection_dir) if f.endswith(".png")}
        common_names  = sorted(list(ct_names.intersection(infect_names)))
        
        diff = len(ct_names) + len(infect_names) - 2*len(common_names)
        if diff > 0:
            logger.warning(f"Dropped {diff} mismatched files. Using {len(common_names)} aligned pairs.")
        
        self.ct_files     = [f"{n}.png" for n in common_names]
        self.infect_files = [f"{n}.png" for n in common_names]
    
    def get_total_count(self) -> int:
        return len(self.ct_files)

    def get_data_generator(self):
        for ct_file, infect_file in tqdm(
            zip(self.ct_files, self.infect_files), 
            total=len(self.ct_files), 
            desc="Processing PNG"
        ):   
            ct_img = cv2.imread(os.path.join(self.ct_dir, ct_file), cv2.IMREAD_UNCHANGED)
            inf_img = cv2.imread(os.path.join(self.infection_dir, infect_file), cv2.IMREAD_UNCHANGED)
            
            ct_vol = np.expand_dims(ct_img, axis=-1)
            inf_vol = np.expand_dims(inf_img, axis=-1)
            lung_vol = None
            yield ct_vol, lung_vol, inf_vol