import os
import sys
import argparse
import numpy as np
import tifffile
from save_numpy_as_tif import save_tiff

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.bce_config import NORM_LOWER_BOUND, NORM_UPPER_BOUND

from configs.bce_config import CROPPING_REGIONS


def clip(img: np.array, down: float, up: float) -> np.array:
    img_clipped = np.clip(img, down, up)
    normalized_img = (img_clipped - down) / (up - down)    
    return normalized_img

""" def clip(img: np.array, down: float, up: float) -> np.array:
    c = img * (img >= down) * (img <= up)
    return (c - down) / (up - down) """


def crop(img: np.array, region_name: str) -> np.array:
    if region_name not in CROPPING_REGIONS:
        raise ValueError(f"Region '{region_name}' not found in cropping regions.")
    
    x1, y1 = CROPPING_REGIONS[region_name]['top_left']   
    x2, y2 = CROPPING_REGIONS[region_name]['bottom_right']

    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        raise ValueError("Cropping coordinates are out of bounds.")
    
    return img[y1:y2, x1:x2]

def main():
    """
    Main function to process images from a directory, normalize them, and save them as TIFF files.
    """
    args = parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    down = NORM_LOWER_BOUND #args.down
    up = NORM_UPPER_BOUND   #args.up

    images_np = []
    filenames = []

    cropped = []

    try:
        files = sorted(os.listdir(data_dir), key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    except FileNotFoundError as e:
        print(f"Error: Directory {data_dir} not found.")
        return
    
    start_index = 0
    for index, file in enumerate(files):
        file_path = os.path.join(data_dir, file)
        try:
            image = tifffile.imread(file_path)
            images_np.append(clip(image, down, up))
            filenames.append(f'reco_{str(index + start_index).zfill(4)}')
            # filenames.append(f'reco_{index + 1}')
            print(file_path, filenames[-1])
        except Exception as e:
            print(f"Error loading {file}: {e}")

    save_tiff(images_np, save_dir, filenames)

    # save_tiff(cropped, r'D:\datasets\луковица\experiments_data\roi', filenames)

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Convert numpy images to TIFF files.")
    parser.add_argument('--data_dir', type=str, help='Directory containing the numpy files.')
    parser.add_argument('--save_dir', type=str, help='Directory to save the TIFF files.')
    #parser.add_argument('--down', type=float, default=0.0005, help='Lower bound for normalization.')
    #parser.add_argument('--up', type=float, default=0.0017, help='Upper bound for normalization.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
