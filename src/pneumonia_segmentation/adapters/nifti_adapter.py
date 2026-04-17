import os, nibabel as nib
from tqdm import tqdm
from pneumonia_segmentation import logging
from pneumonia_segmentation.adapters import BaseDataSourceAdapter

class NiftiAdapter(BaseDataSourceAdapter):
    
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
        
        self._validate_structure()
        
        self.ct_files     = sorted([f for f in os.listdir(self.ct_dir) if f.endswith(".nii")])
        self.lung_files   = sorted([f for f in os.listdir(self.lung_mask_dir) if f.endswith(".nii")])
        self.infect_files = sorted([f for f in os.listdir(self.infection_dir) if f.endswith(".nii")])
        
        self._validate_counts()
        
    def _validate_structure(self):
        missing = [
            f"  - {subdir}/  ({desc})"
            for subdir, desc in self.REQUIRED_SUBDIRS.items()
            if not os.path.isdir(os.path.join(self.data_dir, subdir))
        ]
        if missing:
            raise FileNotFoundError(
                f"NiftiAdapter expects at minimum:\n"
                f"  data_dir/\n"
                f"  ├── ct_scans/\n"
                f"  └── infection_mask/\n\n"
                f"Missing folders:\n" + "\n".join(missing)
            )
            
        self.has_lung_mask = os.path.isdir(os.path.join(self.data_dir, "lung_mask"))
        if not self.has_lung_mask:
            logging.warning("lung_mask folder is not found — use full_slice instead of crop_to_lung_roi")

    def _validate_counts(self):
        counts = {
            "ct_scans":       len(self.ct_files),
            "lung_mask":      len(self.lung_files),
            "infection_mask": len(self.infect_files),
        }
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"Mismatch in .nii file counts across folders: {counts}\n"
                f"Each folder must contain the same number of files."
            ) 
    
    def get_total_count(self) -> int:
        return len(self.ct_files)

    def get_data_generator(self):
        for ct_file, lung_file, infect_file in tqdm(
            zip(self.ct_files, self.lung_files, self.infect_files), 
            total=len(self.ct_files), 
            desc="Processing NIfTI → PNG"
        ):
            ct_vol   = nib.load(os.path.join(self.ct_dir,        ct_file)).get_fdata()
            inf_vol  = nib.load(os.path.join(self.infection_dir, infect_file)).get_fdata()
            lung_vol = nib.load(os.path.join(self.lung_mask_dir, lung_file)).get_fdata() \
                            if self.has_lung_mask else None
            
            yield ct_vol, lung_vol, inf_vol