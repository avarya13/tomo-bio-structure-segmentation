import os
import sys
import argparse
import importlib
import numpy as np
from PIL import Image  
import tifffile  
import cv2
from save_numpy_as_tif import save_tiff
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

# from configs.layers_rnd import config.CROPPING_REGIONS


def crop(img: np.array, region_name: str, config) -> np.array:
    if region_name not in config.CROPPING_REGIONS:
        raise ValueError(f"Region '{region_name}' not found in cropping regions.")
    
    x1, y1 = config.CROPPING_REGIONS[region_name]['top_left']   #47, 47 
    x2, y2 = config.CROPPING_REGIONS[region_name]['bottom_right'] # 2351, 2351 

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        raise ValueError("Cropping coordinates are out of bounds.")
    
    return img[y1:y2, x1:x2]

def save_image(image: np.array, save_dir: str, mode: str, filename: str):
    """Save the image as TIFF or PNG based on mode."""
    if mode == 'image':
        image = image.astype(np.float32)
        save_tiff([image], save_dir, [filename])
    elif mode == 'mask':
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(image)
        img.save(os.path.join(save_dir, f'{filename}.png'))

def load_image(file_path: str, mode: str) -> np.array:
    """Load an image based on the mode."""
    if mode == 'image':
        return tifffile.imread(file_path)
    elif mode == 'mask':
        with Image.open(file_path) as img:
            return np.array(img) / 255.0  
    else:
        raise ValueError(f"Unknown mode '{mode}'")

def process_files(files, data_dir, save_dir, mode, region, start_index, config):
    data_np = []
    filenames = []

    for index, file in enumerate(files):
        file_path = os.path.join(data_dir, file)
        try:
            data = load_image(file_path, mode)
            height, width = data.shape[:2]
            data = cv2.resize(data, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
                
            #TODO: add arg for resizing

            cropped_data = crop(data, region, config)

            data_np.append(cropped_data)
            filenames.append(os.path.splitext(os.path.basename(file_path))[0])
            # filenames.append(f"{'no_ori_' if region == 'center' else 'bu_ori_'}{str(start_index + index).zfill(4)}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    for index, data in enumerate(data_np):
        filename = filenames[index]
        save_image(data, save_dir, mode, filename)

def main():
    args = parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir
    mode = args.mode

    try:
        config = importlib.import_module(args.config)
    except ImportError:
        print(f"Configuration module '{args.config}_config' not found in 'configs' directory.")
        sys.exit(1)

    os.makedirs(save_dir, exist_ok=True)

    try:
        files = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"Error: Directory {data_dir} not found.")
        return

    start_index = 600
    process_files(files, data_dir, save_dir, mode, args.region, start_index, config)

def parse_args():
    parser = argparse.ArgumentParser(description="Crop a region of interest (ROI) from images or masks and save them as TIFF or PNG files.")    
    parser.add_argument('--data_dir', type=str, help='Directory containing the image or mask files to be processed.')
    parser.add_argument('--save_dir', type=str, help='Directory where cropped images or masks will be saved.')
    parser.add_argument('--region', type=str, choices=['top-right', 'center'], default='top-right', 
                        help='Region from which the ROI will be extracted. Options are "top-right" or "center".')
    parser.add_argument('--mode', type=str, choices=['image', 'mask'], default='image', 
                        help="Mode for processing: 'image' for TIFF output, 'mask' for PNG output.")
    parser.add_argument('--config', type=str, default='bce_config', 
                        help='Configuration module name.')
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
