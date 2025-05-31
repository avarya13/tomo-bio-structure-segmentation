import os
import sys
import argparse
import torch
import cv2
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.bce_config import GLOMERULI_CONFIG, LAYERS_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description="Process PNG images and save them as TIFF files.")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the input PNG files.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--mask_type', type=str, choices=['layer', 'glomerulus'], default='glomerulus', help='Type of mask configuration to use.')
    parser.add_argument('--classes', type=int, choices=[2, 3], default=2, help='Number of classes for segmentation (2 or 3).')
    return parser.parse_args()

def create_directories(base_dir):
    dirs = {
        'blurred': os.path.join(base_dir, 'blurred'),
        'thresholded': os.path.join(base_dir, 'thresholded'),
        'visualization': os.path.join(base_dir, 'visualization'),
        'fwd': os.path.join(base_dir, 'fwd'),
        'bwd': os.path.join(base_dir, 'bwd'),
        'interpolated': os.path.join(base_dir, 'interpolated')
    }
    
    for dir in dirs.values():
        os.makedirs(dir, exist_ok=True)    
    return dirs

def process_images(image, kernel_size, sigma, num_conv, filename, dirs, prefix, threshold):
    output_dir = os.path.join(dirs[prefix])
    processed_images = []

    Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(os.path.join(output_dir, f'{prefix}_original_{filename}.png'))
    
    for i in range(num_conv):
        current_image = processed_images[-1].numpy() if processed_images else image.numpy()
        binary_image = (current_image > threshold).astype(np.float32) 
        blurred_image = cv2.GaussianBlur(binary_image, (kernel_size, kernel_size), sigma)
        processed_images.append(torch.Tensor(blurred_image))
       
        #Image.fromarray((blurred_image * 255).astype(np.uint8)).save(os.path.join(output_dir, f'{prefix}_blurred_{filename}_level_{i+1}.png'))
    
    return processed_images

def get_colored_contours(image, color):
    image = np.array(image)
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    edge_img = dilated_img - image

    colored_contours = np.zeros((edge_img.shape[0], edge_img.shape[1], 3), dtype=np.uint8)
    colored_contours[edge_img > 0] = color  
    return colored_contours

def alpha_blend_images(img, alphas, img1_contours, img2_contours):
    img = np.array(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img1_contours = img1_contours / 255.0
    if img2_contours is not None:
        img2_contours = img2_contours / 255.0
        blended_img = alphas[0] * img + alphas[1] * img1_contours + alphas[2] * img2_contours
    else:
        blended_img = alphas[0] * img + alphas[1] * img1_contours
    blended_img = np.clip(blended_img, 0, 1) 
    blended_img = (blended_img * 255).astype(np.uint8)  
    return blended_img

def save_images(dirs, base_name, index, img, final_mask, overlayed):
    filename = f'{base_name}{str(index).zfill(6)}'
    
    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
    final_mask_pil = Image.fromarray(final_mask, mode='L')
    overlayed_pil = Image.fromarray(overlayed)
    
    img_pil.save(os.path.join(dirs['blurred'], f'{filename}.png'))
    final_mask_pil.save(os.path.join(dirs['thresholded'], f'{filename}.png'))
    overlayed_pil.save(os.path.join(dirs['visualization'], f'{filename}.png'))

def load_png_as_tensor(file_path):
    image = Image.open(file_path).convert('L')
    image = np.array(image) / 255.0
    return torch.Tensor(image)

def create_final_mask(img, threshold_low, threshold_high=None, classes=2):
    final_mask = np.zeros(img.shape, dtype=np.uint8)
    
    if classes == 2:
        final_mask[img > threshold_low] = 255
    elif classes == 3:
        final_mask[img > threshold_high] = 255
        final_mask[(img > threshold_low) & (img <= threshold_high)] = 128
    
    return final_mask

def save_image(image, directory, filename):
    image_np = image.cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    image_pil.save(os.path.join(directory, filename))

def process_images_pair(img1_path, img2_path, dirs, threshold_low, threshold_high, num_conv, kernel_size, sigma, classes, device):    
    img1 = load_png_as_tensor(img1_path).to(device)
    img2 = load_png_as_tensor(img2_path).to(device)

    filename_1 = os.path.splitext(os.path.basename(img1_path))[0]
    filename_2 = os.path.splitext(os.path.basename(img2_path))[0]

    processed_images1 = process_images(img1, kernel_size, sigma, num_conv, filename_1, dirs, 'fwd', threshold_low)
    processed_images2 = process_images(img2, kernel_size, sigma, num_conv, filename_2, dirs, 'bwd', threshold_low)
    
    interpolated_images = []

    for i in range(len(processed_images1)): 
        interpolated_image = np.maximum(processed_images1[i], processed_images2[len(processed_images1)-i-1])
        interpolated_images.append(interpolated_image)
        save_image(processed_images1[i], dirs['fwd'], f'fwd_{filename_1}_level_{i}.png')
        save_image(processed_images2[len(processed_images1)-i-1], dirs['bwd'], f'bwd_{filename_2}_level_{i}.png')
        save_image(interpolated_image, dirs['interpolated'], f'inter_{filename_1}_{filename_2}_level_{i}.png')
    
    combined_images = [
        img1,  
        torch.max(interpolated_images[0], interpolated_images[1]), 
        torch.max(interpolated_images[2], interpolated_images[3]),  
        torch.max(interpolated_images[4], interpolated_images[5]),  
        torch.max(interpolated_images[-1], interpolated_images[-2]),  
        img2  
    ]

    img1_contours = get_colored_contours(img1, [0, 255, 0])
    img2_contours = get_colored_contours(img2, [0, 0, 255])

    filename = os.path.splitext(os.path.basename(img1_path))[0]
    base_name = 'reco_' #filename[:-2]
    base_index = int(filename[-6:])

    for index, img in enumerate(combined_images):
        img = img.numpy()
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

        final_mask = create_final_mask(img, threshold_low, threshold_high, classes)

        overlayed = alpha_blend_images(final_mask, [1.0, 0.8, 0.8], img1_contours, img2_contours)
        
        save_images(dirs, base_name, base_index + index, img, final_mask, overlayed)


def process_last_images(last_img_path, dirs, threshold_low, threshold_high, num_conv, kernel_size, sigma, classes, device):
    img = load_png_as_tensor(last_img_path).to(device)
    filename = os.path.splitext(os.path.basename(last_img_path))[0]

    processed_images = process_images(img, kernel_size, sigma, num_conv, filename, dirs, 'fwd', threshold_low)
    combined_images = [
        torch.max(processed_images[0], processed_images[1]), 
        torch.max(processed_images[2], processed_images[3]),  
        torch.max(processed_images[4], processed_images[5]),  
        torch.max(processed_images[-1], processed_images[-2]), 
    ]    

    img_contours = get_colored_contours(img, [0, 255, 0])

    filename = os.path.splitext(os.path.basename(last_img_path))[0]
    base_name = 'reco_' 
    base_index = int(filename[-6:]) + 1

    for index, img in enumerate(combined_images):
        img = img.numpy()
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

        final_mask = create_final_mask(img, threshold_low, threshold_high, classes)

        overlayed = alpha_blend_images(final_mask, [1.0, 0.8, 0.8], img_contours, None)
        
        save_images(dirs, base_name, base_index + index, img, final_mask, overlayed)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Performing the interpolation procedure on the {str(device).upper()}.')

    config = LAYERS_CONFIG if args.mask_type == 'layer' else GLOMERULI_CONFIG

    kernel_size = config['KERNEL_SIZE']
    num_conv = config['NUM_CONV_LAYERS']
    sigma = config['SIGMA']

    data_dir = args.data_dir
    dirs = create_directories(args.save_dir)
    threshold_low = config['THRESHOLD_LOW']
    threshold_high = config.get('THRESHOLD_HIGH', None)

    files = sorted([file for file in os.listdir(data_dir) if file.endswith('png')])

    for i in range(len(files) - 1):
        img1_path = os.path.join(data_dir, files[i])
        img2_path = os.path.join(data_dir, files[i + 1])
        process_images_pair(img1_path, img2_path, dirs, threshold_low, threshold_high, num_conv, kernel_size, sigma, args.classes, device)
    
    #last_img_path = os.path.join(data_dir, files[-1])

    #process_last_images(last_img_path, dirs, threshold_low, threshold_high, num_conv, kernel_size, sigma, args.classes, device)

    print("Image processing complete.")

if __name__ == "__main__":
    main()
