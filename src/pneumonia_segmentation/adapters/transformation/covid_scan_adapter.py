import os, nibabel as nib
from tqdm import tqdm
from pneumonia_segmentation import logging
from pneumonia_segmentation.adapters.transformation import BaseDataTransformationAdapter

class CovidScanAdapter(BaseDataTransformationAdapter):
    REQUIRED_SUBDIRS = {
        "ct_scans":       "CT scan volumes",
        "infection_mask": "Infection segmentation masks",
    }
    OPTIONAL_SUBDIRS = {
        "lung_mask": "Lung segmentation masks (optional)",
    }
    
    def __init__(self, data_dir: str):
        self.data_dir       = data_dir
        self.ct_dir         = os.path.join(data_dir, "ct_scans")
        self.lung_mask_dir  = os.path.join(data_dir, "lung_mask")
        self.infection_dir  = os.path.join(data_dir, "infection_mask")
        self.validate_and_init()
    
    def align_datasets(self):
        self.ct_files     = sorted([f for f in os.listdir(self.ct_dir)        if f.endswith(".nii")])
        self.lung_files   = sorted([f for f in os.listdir(self.lung_mask_dir) if f.endswith(".nii")])
        self.infect_files = sorted([f for f in os.listdir(self.infection_dir) if f.endswith(".nii")])
        self._validate_counts()

    def _validate_counts(self):
        counts = {"ct": len(self.ct_files), "lung": len(self.lung_files), "infect": len(self.infect_files)}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Mismatch in .nii file counts: {counts}")
    
    def get_total_count(self) -> int:
        return len(self.ct_files)

    def get_data_generator(self):
        for ct_file, lung_file, infect_file in tqdm(
            zip(self.ct_files, self.lung_files, self.infect_files), 
            total=len(self.ct_files), 
            desc="Processing NIfTI -> PNG"
        ):
            ct_vol   = nib.load(os.path.join(self.ct_dir,        ct_file)).get_fdata()
            inf_vol  = nib.load(os.path.join(self.infection_dir, infect_file)).get_fdata()
            lung_vol = nib.load(os.path.join(self.lung_mask_dir, lung_file)).get_fdata() \
                            if self.has_lung_mask else None
            
            yield ct_vol, lung_vol, inf_vol