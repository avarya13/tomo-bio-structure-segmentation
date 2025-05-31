import os
import sys
import numpy as np
import cv2
import importlib
from PIL import Image
import tifffile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

# from src.file_utils import ensure_directory_exists

def alpha_blend_images(images, alphas):
    """Blend multiple images using specified alpha coefficients.

    Args:
        images (list): List of NumPy images to blend.
        alphas (list): List of alpha coefficients for blending.
        logger (logging.Logger): Logger for logging errors.

    Returns:
        np.ndarray: Blended image.
    
    Raises:
        ValueError: If the number of images and alpha coefficients do not match.
    """
    if len(images) != len(alphas):
        logger.error("The number of images and alpha coefficients must be the same.")
        raise ValueError("The number of images and alpha coefficients must be the same.")
    blended_img = np.zeros_like(images[0], dtype=np.float32)
    for img, alpha in zip(images, alphas):
        blended_img += alpha * img
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    return blended_img

def apply_morphological_edge_detection(image):
    """Apply morphological edge detection to an image.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Image with edges detected.
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    edge_img = dilated_img - image
    return (edge_img).astype(np.uint8)

def save_image(output_path, filename, image_np):
    """Save a NumPy image to a specified path.

    Args:
        output_path (str): Directory path to save the image.
        filename (str): Filename for the saved image.
        image_np (np.ndarray): Image to save.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If the image fails to save.
    """
    # ensure_directory_exists(output_path)
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    try:
        image_pil = Image.fromarray(image_np)
        image_pil.save(full_path)
    except Exception as e:
        print(f"Failed to save image to {full_path}. Exception: {e}")


def get_color_masks(mask):
    colors = [(0, 255, 0),     
              (0, 0, 255),     
              (255, 255, 0),   
              (255, 0, 0),
            #   (128, 0, 128),   
              (50, 240, 240)]    

    classes = [69, 109, 153, 204, 255]
    print(np.unique(mask))

    output_dir = 'classes'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colored_edge_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for i, cls in enumerate(classes):
        cls_mask = (mask == cls).astype(np.uint8)
        edge_mask = apply_morphological_edge_detection(cls_mask)

        colored_edge_mask[edge_mask > 0] = colors[i]
        cv2.imwrite(os.path.join(output_dir, f'class_{cls}_mask.png'), colored_edge_mask)


    return colored_edge_mask


def overlay_mask_edges(original, mask, inst_mask, output_dir, filename_base, config):
    try:
        edge_mask =  get_color_masks(mask)

        ignore_mask = ((mask == 127) | (mask == 50)).astype(np.uint8) * 255
        ignore_mask = cv2.cvtColor(ignore_mask, cv2.COLOR_GRAY2BGR)  

        rgb_mask = cv2.cvtColor(inst_mask, cv2.COLOR_GRAY2BGR)
        # rgb_mask[ignore_mask == 255] = (255, 0, 0) 

        rgb_mask[inst_mask == 0] = (255, 0, 0) 
        # print('rgb_mask', rgb_mask.shape)

        if len(original.shape) == 2:  
            original_color_np = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) 
        else:
            original_color_np = original  

        # colored_edge_mask = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 3), dtype=np.uint8)

        # colored_edge_mask[edge_mask > 0] = config.TARGETS_COLOR

        colored_edge_overlayed = alpha_blend_images([original_color_np, edge_mask, rgb_mask, ignore_mask], [1.0, 1.0, 0.3, 0.3])
        # colored_edge_overlayed = alpha_blend_images([original_color_np, edge_mask], [1.0, 0.8])
       
        save_image(output_dir, f"{filename_base}.png", colored_edge_overlayed)

    except Exception as e:
        print(f"Failed to create and save edges overlay for {filename_base}. Exception: {e}")

def main():
    config = importlib.import_module("ave_half_seq")
    # mask_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data/data_ave_all/test/y'
    # img_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data/data_ave_all/test/x'
    # inst_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/instability'
    # output_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/ave_overlayed_'  

    mask_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_800/masks'
    img_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_800/slices'
    pred_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_800/predictions'
    inst_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/weights/_confi_right.tif'
    output_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_800/ave_overlayed_right'  
    inst = tifffile.imread(inst_dir)
    # for img_filename, mask_filename, inst_filename in (zip(sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir)), sorted(os.listdir(inst_dir)))):
    for i, (img_filename, mask_filename, pred_filename) in enumerate(zip(sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir)), sorted(os.listdir(pred_dir)))):
        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        # inst_path = os.path.join(inst_dir, inst_filename)
        
        if img_path.endswith('.tif'):
            img = tifffile.imread(img_path) * 255
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #TODO: make inst_mask optional
        # inst_mask =  cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # cv2.imread(inst_path, cv2.IMREAD_GRAYSCALE)

        inst_mask = (inst[i] > 0.5).astype(np.uint8)
        cv2.imwrite('inst_mask.png', inst_mask*255)
        print('inst_mask', inst_mask.shape)
        print(np.unique(inst_mask))

        overlay_mask_edges(img, mask, inst_mask, output_dir, img_filename.split('.')[0], config)


if __name__ == "__main__":
    main()