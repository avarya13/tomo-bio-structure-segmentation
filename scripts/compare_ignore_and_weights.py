import os
import numpy as np
import tifffile
from PIL import Image
import cv2


def alpha_overlay(mask, weight, alpha=0.25, threshold=1):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask
    weight_mask = weight < threshold
    weight_norm = cv2.normalize(weight * weight_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(weight_norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(mask_rgb, 1,  np.stack([weight_norm]*3, axis=-1), alpha, 0)
    return overlay

def apply_weights_to_masks(tiff_path, masks_dir, output_dir, alpha=0.5):
    weights_stack = tifffile.imread(tiff_path)
    os.makedirs(output_dir, exist_ok=True)

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith('.png'))
    
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(masks_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)

        if i >= len(weights_stack):
            continue

        mask = np.array(Image.open(mask_path).convert('L'))  
        mask[mask==255]=230
        weight = weights_stack[i]

        overlay = alpha_overlay(mask, weight, alpha=alpha)
        Image.fromarray(overlay).save(output_path)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tiff_path = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/weights/weights_864_864.tif"
    masks_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/slices_full_enh_cropped/y"
    output_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/weight_vs_ignore"

    apply_weights_to_masks(tiff_path, masks_dir, output_dir, alpha=0.5)