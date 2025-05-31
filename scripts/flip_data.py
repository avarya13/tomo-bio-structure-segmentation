import os
import argparse
import tifffile
from PIL import Image
import numpy as np

def copy_and_augment_files(src_dir, dest_dir, files, prefix, folder_type):
    os.makedirs(dest_dir, exist_ok=True)

    for file in files:
        src_file_path = os.path.join(src_dir, file)
        
        if folder_type == 'x' and file.endswith('.tif'):
            img = tifffile.imread(src_file_path)
            save_func = tifffile.imwrite

        elif folder_type == 'y' and file.endswith('.png'):
            img = np.array(Image.open(src_file_path))  
            save_func = lambda path, data: Image.fromarray(data).save(path)

        else:
            print(f"Skipping unsupported file: {file}")
            continue
        
        new_filename = file.replace('ori', prefix)
        dest_file_path = os.path.join(dest_dir, new_filename)
        save_func(dest_file_path, img)
        
        if prefix == 'ver' or prefix == 'dbl':
            ver_img = img[::-1, ...]  
            ver_file_path = dest_file_path.replace(prefix, 'ver')
            save_func(ver_file_path, ver_img)
        
        if prefix == 'hor' or prefix == 'dbl':
            hor_img = img[:, ::-1, ...]  
            hor_file_path = dest_file_path.replace(prefix, 'hor')
            save_func(hor_file_path, hor_img)
        
        if prefix == 'dbl':
            dbl_img = ver_img[:, ::-1, ...]  
            dbl_file_path = dest_file_path.replace(prefix, 'dbl')
            save_func(dbl_file_path, dbl_img)

def main(args):
    data_dir = args.data_dir
    
    folders = ['x', 'y']  
    
    for mode in ['val', 'test']:
        for folder in folders:
            folder_path = os.path.join(data_dir, mode, folder)
            test_data = sorted(os.listdir(folder_path))

            copy_and_augment_files(folder_path, folder_path, test_data, 'ver', folder)
            copy_and_augment_files(folder_path, folder_path, test_data, 'hor', folder)
            copy_and_augment_files(folder_path, folder_path, test_data, 'dbl', folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy and augment image files from source to destination.")
    
    parser.add_argument('--data_dir', type=str, default=r'data-100-200-color',
                        help='Path to the source data directory.')
    
    args = parser.parse_args()
    main(args)
