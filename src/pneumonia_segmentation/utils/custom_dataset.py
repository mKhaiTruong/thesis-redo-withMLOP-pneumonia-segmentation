import numpy as np
from torch.utils.data import Dataset
from pneumonia_segmentation.utils.augmentation import *

class CustomDataset(Dataset): 
    def __init__(self, images, is_train=False, masks=None, 
                 image_size=512, cache_size=512):
        
        if is_train and masks is None:
            raise ValueError("Training dataset requires masks, but masks=None was provided.")
        
        self.img_paths = images
        self.msk_paths = masks
        self.image_size = image_size
        
        self._cache = {}
        self._cache_size = cache_size
        self.transforms  = get_train_aug(self.image_size) if is_train else get_base_aug(self.image_size)
        
        # Sanity check
        assert len(self.img_paths) > 0, "No images found"
        if masks is not None:
            assert len(self.img_paths) == len(self.msk_paths), \
                f"Number of images ({len(self.img_paths)}) != number of masks ({len(self.msk_paths)})"
    
    def _read_image(self, img_pth):
        img = cv2.imread(img_pth, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_pth}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return img
    
    def _read_mask(self, msk_pth):
        msk = cv2.imread(msk_pth, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise ValueError(f"Cannot read mask: {msk_pth}")
        msk = cv2.resize(msk, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Binary mask 0/1
        msk = (msk > 0).astype(np.uint8)
        return msk
    
    def __getitem__(self, index):
        img_pth = self.img_paths[index]
        msk_pth = self.msk_paths[index] if self.msk_paths else None
        
        # --- cache ---
        if index in self._cache:
            img = self._cache[index]['img']
            msk = self._cache[index].get('mask', None)
        else:
            img = self._read_image(img_pth)
            msk = self._read_mask(msk_pth) if msk_pth else None
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[index] = {'img': img, 'mask': msk}
        
        # --- augmentation ---
        if msk is not None:
            aug = self.transforms(image=img, mask=msk)
            img_t, msk_t = aug['image'], aug['mask'].long()
        else:
            aug = self.transforms(image=img)
            img_t, msk_t = aug['image'], None
        
        # ---- OUTPUT ----
        output = {'image': img_t}
        
        if msk_t is not None:
            output['mask'] = msk_t.unsqueeze(0)
        return output

    def __len__(self): return len(self.img_paths) 