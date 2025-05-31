import os
import argparse
import numpy as np
import tifffile
from shutil import copyfile


def stack_image_slices(src_dir, dest_dir):
    """
    Creates three-channel images by stacking the current slice with its neighboring slices.
    
    For edge slices, the adjacent slice is repeated on both sides.

    Parameters:
        src_dir (str): Path to the source directory containing .tif images.
        dest_dir (str): Path to the destination directory where processed images will be saved.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    images_paths = [os.path.join(src_dir, filename) for filename in os.listdir(src_dir) if filename.endswith('.tif')]

    if not images_paths:
        print(f"No image files found in the directory {src_dir}.")
        return

    for index in range(len(images_paths)):
        if index > 0 and index < len(images_paths) - 1:
            # For non-edge slices, use the previous and next slices
            prev_index = index - 1
            next_index = index + 1
        elif index == 0:
            # For the first slice, use the next slice for both neighbors
            next_index = index + 1
            prev_index = next_index
        elif index == len(images_paths) - 1:
            # For the last slice, use the previous slice for both neighbors
            prev_index = index - 1
            next_index = prev_index

        prev_image = tifffile.imread(images_paths[prev_index])
        cur_image = tifffile.imread(images_paths[index])
        next_image = tifffile.imread(images_paths[next_index])

        stacked_image = np.stack([prev_image, cur_image, next_image], axis=-1)

        save_path = os.path.join(dest_dir, os.path.basename(images_paths[index]))
        #tifffile.imwrite(save_path, stacked_image, compression=None)
        tifffile.imwrite(save_path, stacked_image)



def copy_mask_images(src_dir, dest_dir):
    """
    Copies mask images (.png files) from the source directory to the destination directory.

    Parameters:
        src_dir (str): Path to the source directory containing .png mask images.
        dest_dir (str): Path to the destination directory where mask images will be saved.
    """
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(src_dir):
        if filename.endswith('.png'):
            src_file_path = os.path.join(src_dir, filename)
            dest_file_path = os.path.join(dest_dir, filename)
            copyfile(src_file_path, dest_file_path)


def check_directory_structure(data_dir):
    """
    Checks if the expected directory structure exists in the specified data directory.

    Parameters:
        data_dir (str): Path to the source directory to check.

    Returns:
        bool: True if the structure is valid, False otherwise.
    """
    expected_structure = [
        os.path.join(data_dir, 'bu', 'x'),
        os.path.join(data_dir, 'bu', 'y'),
        os.path.join(data_dir, 'no', 'x'),
        os.path.join(data_dir, 'no', 'y')
    ]
    return all(os.path.isdir(path) for path in expected_structure)


def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir

    if not check_directory_structure(data_dir):
        print("Invalid directory structure. Please ensure the following directories exist:")
        print("src_dir/bu/x/, src_dir/bu/y/, src_dir/no/x/, src_dir/no/y/")
        return

    subdirs = {
        'bu/x': stack_image_slices,
        'no/x': stack_image_slices,
        'bu/y': copy_mask_images,
        'no/y': copy_mask_images
    }

    for subdir, func in subdirs.items():
        src_subdir = os.path.join(data_dir, subdir)
        dest_subdir = os.path.join(save_dir, subdir)
        
        func(src_subdir, dest_subdir)

    print(f"Processed images and copied mask images saved to {save_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and copy images based on directory type.")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the source directory with the images.')
    parser.add_argument('--save_dir', type=str, required=True, 
                        help='Directory to save the processed images.')

    args = parser.parse_args()
    main(args)
