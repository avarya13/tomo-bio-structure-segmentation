import os
import sys
import cv2
import numpy as np
from PIL import Image
import tifffile
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

def alpha_blend_images(images, alphas):
    if len(images) != len(alphas):
        raise ValueError("The number of images and alpha coefficients must be the same.")
    blended_img = np.zeros_like(images[0], dtype=np.float32)
    for img, alpha in zip(images, alphas):
        blended_img += alpha * img
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    return blended_img

def apply_morphological_edge_detection(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    edge_img = dilated_img - image
    return edge_img.astype(np.uint8)

def save_image(output_path, filename, image_np):
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    try:
        image_pil = Image.fromarray(image_np)
        image_pil.save(full_path)
    except Exception as e:
        print(f"Failed to save image to {full_path}. Exception: {e}")

def get_color_masks(mask):
    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 0), (50, 240, 240)]
    classes = [69, 109, 153, 204, 255]
    colored_edge_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for i, cls in enumerate(classes):
        cls_mask = (mask == cls).astype(np.uint8)
        edge_mask = apply_morphological_edge_detection(cls_mask)
        colored_edge_mask[edge_mask > 0] = colors[i]

    return colored_edge_mask

def get_prev_mask_edges(prev_mask):
    prev_edges = apply_morphological_edge_detection(prev_mask)
    overlay = np.zeros((*prev_edges.shape, 3), dtype=np.uint8)
    overlay[prev_edges > 0] = (128, 128, 128)  # светло-серый
    return overlay

def overlay_mask_edges(original, mask, pred, inst_mask, change_map, prev_mask, output_dir, filename_base, config):
    try:
        edge_mask = get_color_masks(mask)

        ignore_areas = ((mask == 127) | (mask == 50) | (inst_mask == 1))
        ignore_mask = (ignore_areas & (pred != mask)).astype(np.uint8)
        error_mask = (~ignore_areas & (pred != mask)).astype(np.uint8)

        blue_ignore_mask = np.zeros((*ignore_mask.shape, 3), dtype=np.uint8)
        blue_ignore_mask[ignore_mask > 0] = (0, 0, 255)

        red_error_mask = np.zeros((*error_mask.shape, 3), dtype=np.uint8)
        red_error_mask[error_mask > 0] = (255, 0, 0)

        change_map_color = np.zeros((*error_mask.shape, 3), dtype=np.uint8)
        change_map_color[change_map > 0] = (255, 255, 0)

        if len(original.shape) == 2:
            original_color_np = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            original_color_np = original

        overlays = [
            original_color_np,
            edge_mask,
            # blue_ignore_mask,
            # red_error_mask,
            # change_map_color
        ]
        alphas = [1.0, 1.0] #, 0.4, 0.4, 0.2

        # if prev_mask is not None:
        #     prev_edge_overlay = get_prev_mask_edges(prev_mask)
        #     overlays.append(prev_edge_overlay)
        #     alphas.append(0.5)

        colored_edge_overlayed = alpha_blend_images(overlays, alphas)

        save_image(output_dir, f"{filename_base}.png", colored_edge_overlayed)
        cv2.imwrite(f'{filename_base}_ignore.png', ignore_mask * 255)
        cv2.imwrite(f'{filename_base}_error.png', error_mask * 255)

    except Exception as e:
        print(f"Failed to create and save edges overlay for {filename_base}. Exception: {e}")

def main():
    config = importlib.import_module("ave_half_seq")

    mask_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/20250410-1415/20250410-1415_1000/masks'
    img_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/20250410-1415/20250410-1415_1000/slices'
    pred_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.ave_4n7_140_20_140/20250410-1415/20250410-1415_1000/predictions'
    inst_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/weights/_conf.tif'
    output_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/ave_overlayed'
    change_map_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/change_maps'

    inst = tifffile.imread(inst_dir)
    filenames = sorted(os.listdir(img_dir))

    for i, img_filename in enumerate(filenames):
        future_idx = ((i // 10) + 1) * 10
        if future_idx >= len(filenames):
            future_idx = None  

        img_path = os.path.join(img_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename)
        pred_path = os.path.join(pred_dir, img_filename)

        if img_path.endswith('.tif'):
            img = tifffile.imread(img_path) * 255
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        inst_mask = (inst[i] <= 0.5).astype(np.uint8)

        if future_idx is not None and future_idx < len(filenames):
            future_name = filenames[future_idx]
            change_map_path = os.path.join(change_map_dir, future_name)
            prev_mask_path = os.path.join(mask_dir, future_name)
            change_map = cv2.imread(change_map_path, cv2.IMREAD_GRAYSCALE)
            prev_mask = cv2.imread(prev_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            change_map = np.zeros_like(mask)
            prev_mask = None

        filename_base = os.path.splitext(img_filename)[0]
        overlay_mask_edges(img, mask, pred, inst_mask, change_map, prev_mask, output_dir, filename_base, config)

if __name__ == "__main__":
    main()
