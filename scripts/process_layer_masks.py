import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def process_and_save_masks(layer_masks, layer_value, save_layer_dir):
    """
    Process and save masks.
    """
    results = []
    
    for index, layer_mask in enumerate(layer_masks):
        specific_layer_mask = layer_mask == layer_value
        results.append(specific_layer_mask)
        file_number = 600 + index * 10  
        save_path = os.path.join(save_layer_dir, f"reco_{str(file_number).zfill(6)}.png")  
        Image.fromarray(specific_layer_mask.astype(np.uint8) * 255).save(save_path)  
    return results


def create_and_save_heatmap(data, save_path, cmap='hot'):
    """
    Create and save a heatmap image from the given data.
    """
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize data

    num_colors = 256
    color_map = plt.get_cmap(cmap)(np.linspace(0, 1, num_colors))  # Get colormap
    
    indexed_data = (normalized_data * (num_colors - 1)).astype(np.uint8)  # Map data to colors
    
    heatmap = color_map[indexed_data, :3]  # Extract RGB channels
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8)) 
    heatmap_image.save(save_path) 

def load_images(dir_path):
    """
    Load all PNG images from the specified directory.
    """
    images = []
    
    try:
        files = sorted(os.listdir(dir_path))
    except FileNotFoundError:
        print(f"Error: Directory {dir_path} not found.")
        return images
    
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(dir_path, file)
            try:
                image = np.array(Image.open(file_path).convert('L'))  # Load image as grayscale
                images.append(image)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    print(f"Loaded {len(images)} images from {dir_path}")
    return images



def main():
    """
    Main function to process masks and generate output files.
    """
    args = parse_args()
    save_dir = args.save_dir
    data_dir = args.data_dir
    save_layer_dir = args.save_layer_dir

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_layer_dir, exist_ok=True)

    layer_masks = load_images(data_dir)
   
    processed_masks = process_and_save_masks(layer_masks, 69, save_layer_dir)
    
    binary_processed_masks = [(mask > 0).astype(np.uint8) for mask in processed_masks]
    sum_mask = np.sum(binary_processed_masks, axis=0)


    binary_sum_mask = (sum_mask > 0).astype(np.uint8) * 255


    kernel = np.ones((3,3), np.uint8) 
    closed_binary_sum_mask = cv2.dilate(binary_sum_mask, kernel, iterations=1)
    closed_binary_sum_mask = cv2.erode(binary_sum_mask, kernel, iterations=1) 
        
    # Save summary mask
    summary_mask_path = os.path.join(save_dir, 'summary_mask.png')
    Image.fromarray(binary_sum_mask).convert('L').save(summary_mask_path)

    # Save heatmap
    heatmap_path = os.path.join(save_dir, 'heatmap.png')
    create_and_save_heatmap(sum_mask, heatmap_path)

    # Save binary mask
    binary_mask_path = os.path.join(save_dir, 'binary_mask.png')
    Image.fromarray(closed_binary_sum_mask).convert('L').save(binary_mask_path)
    
    print(f"Binary mask shape: {binary_sum_mask.shape}")


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process and combine masks.")
    parser.add_argument('--data_dir', type=str, help="Directory containing masks with multiple layers.")
    parser.add_argument('--save_layer_dir', type=str, help="Directory to save single layer masks.")
    parser.add_argument('--save_dir', type=str, help='Directory to save the summary mask and the heatmap.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
