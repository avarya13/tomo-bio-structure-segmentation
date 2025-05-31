import os
import cv2
from enum import Enum
import numpy as np
import tifffile
from PIL import Image


class GrayscaleColors(Enum):
    BACKGROUND_COLOR = 0
    IGNORE_INDEX = 127
    GL = 69
    EPL = 109
    MCL = 255
    GCL = 204
    SL = 153

def remove_noise(mask, kernel_size=3):
    denoised = cv2.medianBlur(mask, kernel_size)
    return denoised.astype(np.uint8)

def fill_holes(mask):
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask  


def add_border_px(layer_mask, binary_mask):

    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    border = (dilated - binary_mask) > 0

    result = layer_mask.copy()
    result[border] = 127
    result = fill_holes(result) 
    return result

def get_largest_component(mask, target_class):
    class_mask = (mask == target_class).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=4)  
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)

    return largest_component


def remove_small_components(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    filtered_mask = np.zeros_like(mask, dtype=np.uint8)

    for i in range(1, num_labels):  
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 1  

    return filtered_mask

def check_single_component(mask):
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    return num_labels == 2

def save_components(mask, target_class, output_folder):
    class_mask = (mask == target_class).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=4)

    os.makedirs(output_folder, exist_ok=True)

    for label in range(0, num_labels):  
        component_mask = (labels == label).astype(np.uint8) * 255  
        component_filename = os.path.join(output_folder, f"component_{label}.png")
        cv2.imwrite(component_filename, component_mask)

def prepare_layer(mask, path, color, gcl=False):
    if gcl:
        layer_mask = get_largest_component(mask, color)
    else:
        layer_mask = (mask == color).astype(np.uint8)

    layer_mask = remove_noise(layer_mask)
    inv_layer_mask = 1 - layer_mask  

    inv_layer_mask = remove_small_components(inv_layer_mask, min_size=500)
    layer_mask = (1 - inv_layer_mask)

    layer_mask = remove_small_components(layer_mask, min_size=500)

    cv2.imwrite(path, layer_mask*255)
    return layer_mask

def prepare_bg(mask, path):
    bg_mask = get_largest_component(mask, 0)

    inverted_bg_mask = 1- bg_mask
    cv2.imwrite('inverted_bg_mask.png', inverted_bg_mask*255)

    bg_mask = 1 - get_largest_component(inverted_bg_mask, 1)

    denoised_bg_mask = remove_noise(bg_mask)

    obj_mask = 1 - denoised_bg_mask  

    clean_obj_mask = remove_small_components(obj_mask, min_size=500)

    bg_mask = (1 - clean_obj_mask) 
    bg_mask = remove_small_components(bg_mask, min_size=500)
    cv2.imwrite(path, bg_mask*255)
    return bg_mask

def process_masks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".tif")):
            input_path = os.path.join(input_folder, filename)
            output_name = os.path.splitext(filename)[0] + ".png" 
            refined_output_name = os.path.splitext(filename)[0] + "_refined.png" 
            output_path = os.path.join(output_folder, output_name)
            bg_path = os.path.join('bg', output_name)
            mcl_path = os.path.join('mcl', output_name)
            gl_path = os.path.join('gl', output_name)
            sl_path = os.path.join('sl', output_name)
            epl_path = os.path.join('epl', output_name)
            gcl_path = os.path.join('gcl', output_name)
            refined_color_path = os.path.join('refined_color', refined_output_name)
            refined_path = os.path.join('refined', output_name)
            combined_path = os.path.join('combined', output_name)
            layer_folders = ['bg', 'gcl', 'gl', 'epl', 'mcl', 'sl', 'refined', 'refined_color', 'combined']
            for folder in layer_folders:
                os.makedirs(folder, exist_ok=True)

            if filename.endswith(".tif"):
                mask = tifffile.imread(input_path)
            else:
                mask = np.array(Image.open(input_path))

            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            mask = mask.astype(np.uint8)

            # background mask
            bg_mask = prepare_bg(mask, bg_path)


            # center (GCL) mask
            gl_mask = prepare_layer(mask, gl_path, GrayscaleColors.GL.value)
            epl_mask = prepare_layer(mask, epl_path, GrayscaleColors.EPL.value)
            mcl_mask = prepare_layer(mask, mcl_path, GrayscaleColors.MCL.value)
            gcl_mask = prepare_layer(mask, gcl_path, GrayscaleColors.GCL.value, gcl=True)
            sl_mask = prepare_layer(mask, sl_path, GrayscaleColors.SL.value)

            # combine masks 
            combined_mask = np.zeros_like(mask)
            combined_mask[bg_mask > 0] = GrayscaleColors.BACKGROUND_COLOR.value
            combined_mask[gl_mask > 0] = GrayscaleColors.GL.value
            combined_mask[epl_mask > 0] = GrayscaleColors.EPL.value
            combined_mask[mcl_mask > 0] = GrayscaleColors.MCL.value
            combined_mask[gcl_mask > 0] = GrayscaleColors.GCL.value
            combined_mask[sl_mask > 0] = GrayscaleColors.SL.value
            cv2.imwrite(combined_path, combined_mask)

            # S1
            refined_mask = combined_mask.copy()
            condition = ((bg_mask == 0) & (refined_mask == 0)) | ((gcl_mask == 0) & (refined_mask == 204))
            refined_mask[condition] = 127
            cv2.imwrite(refined_path, refined_mask)

            refined_color_mask = np.stack([refined_mask] * 3, axis=-1)
            refined_color_mask[condition] = [0, 255, 0] 
            cv2.imwrite(refined_color_path, refined_color_mask)





if __name__ == "__main__":
    input_folder = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/full_masks_src"
    # input_folder = r'D:\datasets\луковица\layers_masks\layers_masks'
    output_folder = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/layers_enh_masks_full_ignore"

    process_masks(input_folder, output_folder)