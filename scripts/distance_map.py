import os
import cv2
import numpy as np
import tifffile
import argparse
from tqdm import tqdm


def create_dist_map(mask, i, d=10):
    dist_map = np.zeros(mask.shape, dtype=np.float32)
    classes = np.unique(mask)

    ignore_mask = ((mask == 127) | (mask == 50)).astype(np.uint8)
    mask[(mask == 127) & (mask == 50)] = 0

    for cls in classes:
        class_mask = (mask == cls).astype(np.uint8)
        class_dist_map = cv2.distanceTransform(class_mask, cv2.DIST_L2, 5)
        dist_map += (class_dist_map).astype(np.float32)

    dist_map = np.minimum(dist_map, d)
    dist_map_normalized = dist_map / d

 
def main(mask_dir, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)

    weights = []

    for i, mask_file in tqdm(enumerate(sorted(os.listdir(mask_dir))), ncols=80, desc='Total'):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        resized_mask = mask.copy() #cv2.resize(mask, (864, 864)) #TODO: change hardcoded
        weights.append(create_dist_map(resized_mask, i))
    
    weights = np.array(weights)
    tifffile.imwrite(os.path.join(save_dir, f'{file_name}.tif'), weights.astype(np.float32), compression="zlib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process masks and save weights.')
    parser.add_argument('--mask_dir', type=str, help='Directory containing the mask files.')
    parser.add_argument('--file_name', type=str, help='Name of the output file (without extension).')
    parser.add_argument('--save_dir', type=str, default='weights', help='Directory where weights will be saved (default: weights).')

    args = parser.parse_args()

    main(args.mask_dir, args.save_dir, args.file_name)
