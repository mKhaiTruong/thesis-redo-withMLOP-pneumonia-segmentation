import os, sys
from tqdm.auto import tqdm
from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException

from pneumonia_segmentation.utils.covid_ct_processing import *
from pneumonia_segmentation.entity.entity_config import DataTransformationConfig
from pneumonia_segmentation.adapters.factory import TransformationAdapterFactory

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config   = config
        self.adapters = [
            TransformationAdapterFactory.get_adapter(d["source_type"], d["path"])
            for d in self.config.data_dirs
        ]
    
    def _prepare_output_dirs(self):
        for d in [self.config.out_train_dir, self.config.out_valid_dir, self.config.out_infer_dir]:
            os.makedirs(os.path.join(d, "img"), exist_ok=True)
            os.makedirs(os.path.join(d, "msk"), exist_ok=True)
            
    def transform(self):
        try:
            self._prepare_output_dirs()
            img_counter = 1
            
            for adapter in self.adapters:
                for ct_vol, lung_vol, infect_vol in adapter.get_data_generator():
                    for i in range(0, ct_vol.shape[2], self.config.params_slice_interval):
                        ct_slice     = ct_vol[:, :, i]
                        lung_slice   = lung_vol[:, :, i] if lung_vol is not None else None
                        infect_slice = infect_vol[:, :, i]
                        
                        if should_skip_slice(infect_slice, self.config.params_skip_background_ratio):
                            continue
                            
                        mask_bin = (infect_slice > 0).astype("uint8")
                        
                        if lung_slice is not None:
                            ct_crop, mask_crop = crop_to_lung_roi(ct_slice, mask_bin, lung_slice)
                        else:
                            ct_crop, mask_crop = resize_full_slice(ct_slice, mask_bin, self.config.params_image_size)
                
                        img, mask = normalize_and_colormap(ct_crop, mask_crop, self.config.params_image_size)
                        
                        self._save_pair(img, mask, self._get_output_dir(), img_counter)
                        img_counter += 1
                
                if img_counter > 1:
                    logging.info(f"Data transformation completed: {img_counter - 1} pairs saved")  
        except Exception as e:
            raise CustomException(e, sys)
    
    def _get_output_dir(self) -> str:
        r = random.random()
        if r < self.config.params_infer_size:
            return self.config.out_infer_dir
        elif r < self.config.params_infer_size + self.config.params_valid_size:
            return self.config.out_valid_dir

        return self.config.out_train_dir

    def _save_pair(self, img, mask, out_dir: str, counter: int) -> None:
        img_saved = cv2.imwrite(os.path.join(out_dir, "img", f"scan_{counter:05d}.png"), img)
        mask_saved= cv2.imwrite(os.path.join(out_dir, "msk", f"mask_{counter:05d}.png"), mask)
        
        if not img_saved or not mask_saved:
            logging.warning(f"Failed to save pair {counter}")