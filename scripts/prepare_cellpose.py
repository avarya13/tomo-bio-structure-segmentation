import os
import sys
import argparse
import numpy as np
from PIL import Image
import tifffile
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs.bce_config as config

def crop_center(image, target_height, target_width):
    h, w = image.shape[:2]
    if h < target_height or w < target_width:
        raise ValueError(f"Image size {h}x{w} is smaller than the target size {target_height}x{target_width}.")
    x_center = w // 2
    y_center = h // 2
    x_start = x_center - target_width // 2
    y_start = y_center - target_height // 2
    x_end = x_start + target_width
    y_end = y_start + target_height
    
    return image[y_start:y_end, x_start:x_end]

def save_images_and_masks(images, filenames, save_dir, is_mask=False):
    for index, image in enumerate(images):
        name = filenames[index]
        if is_mask:
            image = Image.fromarray(image).convert('L')
            image.save(os.path.join(save_dir, 'y', f'{name}.png'))
        else:
            image = image.astype(np.float32)
            tifffile.imwrite(os.path.join(save_dir, 'x', f'{name}.tif'), image)

def binarize(mask):
    mask[mask > 0] = 255
    return mask

def normalize_to_range(img: np.array) -> np.array:
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min)

def process_mask_with_conflicts(mask):
    obj_classes = np.unique(mask)
    obj_classes = obj_classes[obj_classes != 0]

    conflict_mask = np.zeros_like(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated_masks = []

    for cls in obj_classes:
        cls_mask = (mask == cls).astype(np.uint8)
        dilated_mask = cv2.dilate(cls_mask, kernel)
        dilated_masks.append(dilated_mask)

    sum_dilated_mask = np.sum(dilated_masks, axis=0)
    conflict_mask[sum_dilated_mask > 1] = 255

    mask[conflict_mask > 0] = 0
    binary_mask = binarize(mask)

    return binary_mask

def main():
    args = parse_args()
    images_dir = os.path.join(args.data_dir, 'x')
    masks_dir = os.path.join(args.data_dir, 'y')
    save_dir = args.save_dir  

    for folder in ['x', 'y']:
        os.makedirs(os.path.join(save_dir, folder), exist_ok=True)

    images = []
    filenames_img = []
    masks = []
    filenames_masks = []

    for image_file in sorted(os.listdir(images_dir)):
        image_path = os.path.join(images_dir, image_file)

        image = np.array(Image.open(image_path).convert('RGB'))  
        if image.ndim > 1:
            image = np.max(image, axis=-1)
            
        filename_img = os.path.splitext(image_file)[0]
        filename_mask = filename_img.replace('img', 'masks')

        mask_path = os.path.join(masks_dir, f"{filename_mask}.png")
        
        mask = np.array(Image.open(mask_path).convert('L'))

        if image.shape[0] >= config.INPUT_HEIGHT and image.shape[1] >= config.INPUT_WIDTH and \
           mask.shape[0] >= config.INPUT_HEIGHT and mask.shape[1] >= config.INPUT_WIDTH:
            
            cropped_image = crop_center(image, config.INPUT_HEIGHT, config.INPUT_WIDTH)
            cropped_mask = crop_center(mask, config.INPUT_HEIGHT, config.INPUT_WIDTH)

            if cropped_image.shape == (config.INPUT_HEIGHT, config.INPUT_WIDTH) and \
               cropped_mask.shape == (config.INPUT_HEIGHT, config.INPUT_WIDTH):
                processed_mask = process_mask_with_conflicts(cropped_mask)
                images.append(cropped_image)
                masks.append(processed_mask)
                filenames_img.append(filename_img)
                filenames_masks.append(filename_mask)

    images = [normalize_to_range(image) for image in images]
    save_images_and_masks(images, filenames_img, save_dir)
    save_images_and_masks(masks, filenames_masks, save_dir, is_mask=True)

    print(f"Total saved images: {len(os.listdir(os.path.join(save_dir, 'x')))}")
    print(f"Total saved masks: {len(os.listdir(os.path.join(save_dir, 'y')))}")

def parse_args():
    parser = argparse.ArgumentParser(description="Select and process dataset.")
    parser.add_argument('-d', '--data_dir', default=r'D:\datasets\cellpose_exp\cell\test', type=str, help="Directory containing source data.")
    parser.add_argument('-s', '--save_dir', default=r'D:\datasets\луковица\experiments_data\cellpose\test', type=str, help="Directory to save output dataset.")
    return parser.parse_args()

if __name__ == "__main__":
    main()
