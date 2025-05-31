import os
import shutil
import numpy as np
import cv2
import tifffile
from PIL import Image

cor_dir = r'D:\datasets\enference.colored_cor\enference.colored_cor'
img_dir = r'D:\datasets\tmp_for_enf'
masks_dir = r'D:\datasets\луковица\experiments_data\60_slices_labeled_dataset_png\y'
dest_dir = r'D:\datasets\луковица\experiments_data\60_masks_cor'
morf_dir = r'D:\datasets\луковица\experiments_data\60_masks_cor_morf'
save_dir = r'D:\datasets\луковица\experiments_data\v2\60_slices_masks_v2'

def create_mask(img, channel):
    diff = img - channel

    marking = np.zeros(img.shape, dtype=np.uint8)
    marking[(diff > 0)] = 255

    kernel = np.ones((3, 3), np.uint8)
    morf_marking = cv2.dilate(marking, kernel)
    morf_marking = cv2.erode(morf_marking, kernel)
    
    return marking


def main():
    os.makedirs(save_dir, exist_ok=True)

    for i, (overlayed_name, img_name) in enumerate(zip(sorted(os.listdir(cor_dir)), sorted(os.listdir(img_dir)))):
    
        overlayed = cv2.imread(os.path.join(cor_dir, overlayed_name), cv2.IMREAD_UNCHANGED)
        img = (tifffile.imread(os.path.join(img_dir, img_name)) * 255).astype(np.uint8)

        r_channel = overlayed[:, :, 2]
                
        #Image.fromarray(marking).save(os.path.join(dest_dir, overlayed_name.replace('_overlayed_colored', '')))
        
        r_channel_mask = create_mask(img, r_channel)

        if i == len(os.listdir(cor_dir)) - 1:
            g_channel = overlayed[:, :, 1]
            g_channel_mask = create_mask(img, g_channel)
            rg_mask = cv2.add(r_channel_mask, g_channel_mask)
            Image.fromarray(rg_mask).save(os.path.join(morf_dir, overlayed_name.replace('_overlayed_colored', '')))
        else:
            Image.fromarray(r_channel_mask).save(os.path.join(morf_dir, overlayed_name.replace('_overlayed_colored', '')))

    Image.fromarray(np.zeros((408, 548), dtype=np.uint8)).save(os.path.join(morf_dir, 'bu_ori_0649.png'))   
    
    morf_files = sorted(os.listdir(morf_dir))
    
    for mask_name, morf_file in zip(sorted(os.listdir(masks_dir)), morf_files):
        print(mask_name, morf_file)
        mask = np.array(Image.open(os.path.join(masks_dir, mask_name)))
        morf_marking = np.array(Image.open(os.path.join(morf_dir, morf_file)))

        new_mask = cv2.add(mask, morf_marking)

        Image.fromarray(new_mask).save(os.path.join(save_dir, mask_name))

    shutil.copyfile(os.path.join(morf_dir, morf_files[-1]), os.path.join(save_dir, 'reco_000900.png'))

if __name__ == "__main__":
    main()
