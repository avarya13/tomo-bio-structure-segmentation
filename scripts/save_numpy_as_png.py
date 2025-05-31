import numpy as np
from PIL import Image
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert binary .npy files to .png format.")
    parser.add_argument('--npy_dir', type=str, required=True, help='Path to the directory containing .npy files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the directory for saving .png files.')
    return parser.parse_args()

def npy_to_png(npy_file_path, png_file_path):
    """Saves a binary .npy file as .png."""
    data = np.load(npy_file_path)
    
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array.")
    
    image_data = (data * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    image.save(png_file_path)
    print(f"Image saved as {png_file_path}")

def convert_directory(npy_dir, output_dir):
    """Converts all binary .npy files to .png in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(npy_dir):
        if file_name.endswith('.npy') or file_name.endswith('.tif'):
            npy_file_path = os.path.join(npy_dir, file_name)
            png_file_name = f'{os.path.splitext(file_name)[0]}.png'
            png_file_path = os.path.join(output_dir, png_file_name)
            npy_to_png(npy_file_path, png_file_path)

if __name__ == "__main__":
    args = parse_arguments()
    convert_directory(args.npy_dir, args.output_dir)
