import os
import argparse
import numpy as np
import tifffile

def save_tiff(images, save_dir, filenames):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for image, filename in zip(images, filenames):
        try:
            tifffile.imwrite(os.path.join(save_dir, f'{filename}.tif'), image)
        except Exception as e:
            print(f"Error saving {filename}: {e}")

def main():
    args = parse_args()
    data_dir = args.data_dir
    save_dir = args.save_dir

    images_np = []
    filenames = []

    try:
        files = sorted(os.listdir(data_dir))
    except FileNotFoundError as e:
        print(f"Error: Directory {data_dir} not found.")
        return

    for file in files:
        if file.endswith('.tif') or file.endswith('.npy'):
            file_path = os.path.join(data_dir, file)
            try:
                image = np.load(file_path)
                images_np.append(image)
                filenames.append(os.path.splitext(file)[0])
            except Exception as e:
                print(f"Error loading {file}: {e}")

    save_tiff(images_np, save_dir, filenames)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert numpy images to TIFF files.")
    parser.add_argument('--data_dir', type=str, help='Directory containing the numpy files.')
    parser.add_argument('--save_dir', type=str, help='Directory to save the TIFF files.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
