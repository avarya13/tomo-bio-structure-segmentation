import os
import sys
import argparse
import numpy as np
from PIL import Image  
import tifffile  
from save_numpy_as_tif import save_tiff
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from configs.bce_config import TOTAL_SLICES, ROI_WIDTH, ROI_HEIGHT


def parse_args():
    parser = argparse.ArgumentParser(description="Generate empty masks for center ROIs.")
    parser.add_argument('--save_dir', type=str, help='Directory to save empty masks.')
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    start_index = 600
    for index in range(TOTAL_SLICES):
        file_name = f'no_ori_{str(start_index + index).zfill(4)}.png'
        mask = Image.fromarray(np.zeros((ROI_HEIGHT, ROI_WIDTH)).astype(np.uint8))
        mask.save(os.path.join(save_dir, file_name))


if __name__ == "__main__":
    main()