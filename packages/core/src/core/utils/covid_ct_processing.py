import sys, random, cv2, numpy as np, nibabel as nib
from core.exception import CustomException

# If there is no lung mask folder...
def resize_full_slice(ct_slice, mask_slice, image_size):
    ct_resized   = cv2.resize(ct_slice.astype("float32"), (image_size, image_size))
    mask_resized = cv2.resize(mask_slice.astype("uint8"),  (image_size, image_size))
    return ct_resized, mask_resized

def load_nii_triplet(ct_path: str, lung_path: str, infect_path: str) -> tuple:
    try:
        ct_volume = nib.load(ct_path).get_fdata()
        lung_volume = nib.load(lung_path).get_fdata()
        infect_volume = nib.load(infect_path).get_fdata()
    
        assert ct_volume.shape == lung_volume.shape == infect_volume.shape, \
            f"Shape mismatch: {ct_volume.shape} vs {lung_volume.shape} vs {infect_volume.shape}"
        
        return ct_volume, lung_volume, infect_volume
    except Exception as e:
        raise CustomException(e, sys)
    
def should_skip_slice(infect_slice: np.ndarray, skip_ratio: float = 0.8) -> bool:
    if not np.any(infect_slice > 0):
        return random.random() > (1 - skip_ratio)
    return False

def crop_to_lung_roi(ct_slice: np.ndarray, mask_bin: np.ndarray, lung_slice: np.ndarray) -> tuple:
    if not np.any(lung_slice > 0):
        return ct_slice, mask_bin
    
    ys, xs = np.where(lung_slice > 0)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    
    lung_area = (y2 - y1) * (x2 - x1)
    img_area = lung_slice.shape[0] * lung_slice.shape[1]
    pad_scale = min(max(lung_area / img_area * 0.3, 0.05), 0.35)

    h_pad = int((y2 - y1) * pad_scale)
    w_pad = int((x2 - x1) * pad_scale)
    
    y1 = max(y1 - h_pad, 0)
    y2 = min(y2 + h_pad, ct_slice.shape[0])
    x1 = max(x1 - w_pad, 0)
    x2 = min(x2 + w_pad, ct_slice.shape[1])
    
    min_h = int(0.5 * ct_slice.shape[0])
    min_w = int(0.5 * ct_slice.shape[1])
    
    if (y2 - y1) < min_h:
        cy = (ys.min() + ys.max()) // 2
        y1 = max(cy - min_h // 2, 0)
        y2 = min(y1 + min_h, ct_slice.shape[0])
    
    if (x2 - x1) < min_w:
        cx = (xs.min() + xs.max()) // 2
        x1 = max(cx - min_w // 2, 0)
        x2 = min(x1 + min_w, ct_slice.shape[1])
    
    return ct_slice[y1:y2, x1:x2], mask_bin[y1:y2, x1:x2]

COLORMAP_MAP = {
    "BONE": cv2.COLORMAP_BONE,
    "JET": cv2.COLORMAP_JET,
}
def normalize_and_colormap(ct_slice: np.ndarray, mask_bin: np.ndarray, image_size: int = 512, colormap: str = "BONE") -> tuple:
    lung_pixels = ct_slice[mask_bin > 0]
    if lung_pixels.size > 0:
        min_val, max_val = lung_pixels.min(), lung_pixels.max()
    else:
        min_val, max_val = ct_slice.min(), ct_slice.max()
    
    ct_norm = np.clip(ct_slice, min_val, max_val)
    ct_norm = ((ct_norm - min_val) / (max_val - min_val + 1e-5)) * 255
    ct_norm = np.nan_to_num(ct_norm, nan=0.0, posinf=255.0, neginf=0.0)
    ct_norm = cv2.resize(ct_norm.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    ct_display = cv2.applyColorMap(ct_norm, COLORMAP_MAP[colormap])
    
    mask_norm = cv2.resize(mask_bin, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    mask_norm = (mask_norm * 255).astype(np.uint8)
    
    return ct_display, mask_norm