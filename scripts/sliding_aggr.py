import os
import sys
import logging
import torch
from torchvision.transforms import functional as F
import segmentation_models_pytorch as smp
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import maximum_filter1d, convolve1d
import tifffile

def load_mask(file_path, device):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
    # image = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  
    mask = np.where(mask == 255, 235, mask)
    return mask 

def load_image(file_path, device):
    image = tifffile.imread(file_path)  
    # image = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  
    return image 

def extract_number(filename):
    basename = os.path.basename(filename)  
    number = ''.join(filter(str.isdigit, basename))  
    return int(number)


img_dir = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/slices_norm" 
# mask_dir = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_roi_clean_enh/y"  
save_img_dir = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/slices_ave_full"  
# save_mask_dir = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data_ave_all_w5" 

os.makedirs(save_img_dir, exist_ok=True)
# os.makedirs(save_mask_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

volume_img = []
volume_masks = []
filenames = []

for img_filename in sorted(os.listdir(img_dir), key=extract_number):
    # print(img_filename)
    img_path = os.path.join(img_dir, img_filename)
    
    img = load_image(img_path, device)

    volume_img.append(img)
    filenames.append(img_filename)

volume_img = np.array(volume_img) # D * H * W

# print('before', volume_img.shape)
# window_size = 5 
# aggr_volume_img = maximum_filter1d(volume_img, size=window_size, axis=0, mode='constant', cval=0)




window_size = 5
pad_width = window_size // 2
kernel = np.ones(window_size) / window_size  

left_pad = volume_img[1: pad_width + 1][::-1] 
right_pad = volume_img[-(pad_width + 1):-1][::-1] 

padded_volume = np.concatenate([left_pad, volume_img, right_pad], axis=0)

aggr_volume_img = convolve1d(padded_volume, kernel, axis=0)

aggr_volume_img = aggr_volume_img[pad_width:-pad_width]


print(f"До усреднения: Min={np.min(volume_img)}, Max={np.max(volume_img)}, Mean={np.mean(volume_img)}, Var={np.var(volume_img)}")
print(f"После усреднения: Min={np.min(aggr_volume_img)}, Max={np.max(aggr_volume_img)}, Mean={np.mean(aggr_volume_img)}, Var={np.var(aggr_volume_img)}")

for i in range(aggr_volume_img.shape[0]):
    tifffile.imwrite(os.path.join(save_img_dir, filenames[i]), aggr_volume_img[i])
