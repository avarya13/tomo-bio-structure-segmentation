import os
import argparse
from PIL import Image
import tifffile as tiff
import cv2

def extract_number(filename):
    number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(number) if number else 0

def halve(img):
    img_left = img[:, :img.shape[1]//2]
    img_right = img[:, img.shape[1]//2:]
    return img_left, img_right

def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if image.ndim == 3:
        cv2.imwrite(output_path, image)
    else:
        tiff.imwrite(output_path, image)

def main():
    parser = argparse.ArgumentParser(description="Crop images in the specified folder.")
    parser.add_argument('--data_dir', type=str, help="Path to the folder containing images.")
    parser.add_argument('--dir_left', type=str, help="Output folder for left halves.")
    parser.add_argument('--dir_right', type=str, help="Output folder for right halves.")
    
    args = parser.parse_args()

    data_dir = args.data_dir
    dir_left_dir = args.dir_left
    dir_right_dir = args.dir_right
    folders = ['x', 'y']  

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        left_folder = os.path.join(dir_left_dir, folder)
        right_folder = os.path.join(dir_right_dir, folder)

        for file in os.listdir(folder_path):
            filename = os.path.join(folder_path, file)
            if filename.endswith('.tif'):
                img = tiff.imread(filename)
            else:
                img = cv2.imread(filename)

            if img.shape[1] == 1616:
                cropped_img = img[:, :1600]  

                img_left, img_right = halve(cropped_img)

                left_filename = os.path.join(left_folder, file)
                right_filename = os.path.join(right_folder, file)
                save_image(img_left, left_filename)
                save_image(img_right, right_filename)

if __name__ == "__main__":
    main()
