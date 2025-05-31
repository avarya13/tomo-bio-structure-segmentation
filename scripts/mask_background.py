import os
import cv2
import tifffile
from normalization import clip

data_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/slices_ave_full'
masks_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/masks.binary/masks'
save_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/slices_ave_full_masked'

NORM_LOWER_BOUND = 0.0005
NORM_UPPER_BOUND = 0.0017

os.makedirs(save_dir, exist_ok=True)

for img_file, mask_file in zip(sorted(os.listdir(data_dir)), sorted(os.listdir(masks_dir))):
    img = tifffile.imread(os.path.join(data_dir, img_file))

    # img = (img - img.min()) / (img.max() - img.min())
    # img = clip(img, NORM_LOWER_BOUND, NORM_UPPER_BOUND)

    mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(img.dtype)

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    # print(masked_img.min(), masked_img.max())

    tifffile.imwrite(os.path.join(save_dir, img_file), masked_img)