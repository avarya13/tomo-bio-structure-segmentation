import os
import cv2
from enum import Enum
import numpy as np
import tifffile
from PIL import Image


def get_largest_component(mask, target_class):
    class_mask = (mask == target_class).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=4)  
    
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)

    return largest_component

def get_class_matrix(matrix, config=None):
    
    mask = np.zeros(matrix.shape, np.float64)
    
    conditions = [
        matrix == 153,
        matrix == 69,
        matrix == 109,
        matrix == 255,
        matrix == 204, 
        matrix == 127,
        matrix == 50
    ]
    
    values = [1, 2, 3, 4, 5, -1, -1]
    
    for condition, value in zip(conditions, values):
        mask[condition] = value
    
    # print(np.unique(mask))
    return mask

class GrayscaleColors(Enum):
    BACKGROUND_COLOR = 0
    IGNORE_INDEX = 127
    IGNORE_INDEX_2 = 50
    GL = 69
    EPL = 109
    MCL = 255
    GCL = 204
    SL = 153

# bg_folder = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/bg'
# input_folder = r"/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/datasets/bulb.layers.png/png"
mask_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/refined'
pred_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_1000/predictions'
img_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250321-1456/20250321-1456_1000/predictions'
# img_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250203-1119/train_visualization/20250203-1119_300/overlayed_edges_colored'

ignore_dir = 'ignore'
refined_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/refined_outer'
refined_color_dir = 'refined_outer_color'
os.makedirs(ignore_dir, exist_ok=True)
os.makedirs(refined_dir, exist_ok=True)
os.makedirs(refined_color_dir, exist_ok=True)

for img_filename in os.listdir(img_dir):
    os.rename(os.path.join(img_dir, img_filename), os.path.join(img_dir, img_filename.replace('_original', '')))

for pred_filename in os.listdir(pred_dir):
    os.rename(os.path.join(pred_dir, pred_filename), os.path.join(pred_dir, pred_filename.replace('_pred', '')))

for img_filename, mask_filename, pred_filename in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(mask_dir)), sorted(os.listdir(pred_dir))):
    # img = cv2.imread(os.path.join(img_dir, img_filename))
    mask = cv2.imread(os.path.join(mask_dir, mask_filename)) #, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    pred = cv2.imread(os.path.join(pred_dir, pred_filename)) #, cv2.IMREAD_GRAYSCALE).astype(np.float64)

    print(img_filename, mask_filename, pred_filename)

    CROPPING_REGIONS = {
    'top-right': {
        'top_left': (280, 747),
        'bottom_right': (1896, 1899)
    },
    'center': {
        'top_left': (865, 1097),
        'bottom_right': (1413, 1505)
    }
    }

    pred = cv2.resize(
        pred,
        dsize=(1616, 1152),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.uint8)

    # img = cv2.resize(
    #     img,
    #     dsize=(1616, 1152),
    #     interpolation=cv2.INTER_CUBIC
    # ).astype(np.uint8)

    original_shape = mask.shape

    region = CROPPING_REGIONS['top-right']

    pred_padded = np.zeros((original_shape[0], original_shape[1], 3), dtype=pred.dtype)

    top_left = region['top_left']
    bottom_right = region['bottom_right']

    pred_padded[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = pred

    pred = pred_padded


    # print(pred.shape, mask.shape, img.shape)
    predicted_classes = get_class_matrix(pred)
    mask_classes = get_class_matrix(mask)

    bulb_mask = (mask != 0).astype(np.uint8)

    ignore_mask = np.zeros_like(mask)
    # ignore_mask[(pred != mask) ] = 1
    ignore_mask[(pred != mask) & (mask != 127)  & ((mask == 0) | (pred == 0)) & (bulb_mask == 0)] = 1 
    cv2.imwrite(os.path.join(ignore_dir, mask_filename.replace('_mask', '')), ignore_mask * 255)

    # refined masks
    
    refined_mask = mask.copy()
    refined_mask[ignore_mask > 0] = GrayscaleColors.IGNORE_INDEX_2.value

    # print((refined_mask == 0).astype(np.uint8), (refined_mask[:, :, 0] == 0).astype(np.uint8).shape)
    bulb_mask = get_largest_component((refined_mask[:, :, 0] == 0).astype(np.uint8), 0)
    cv2.imwrite('bulb_mask.png', (refined_mask[:, :, 0] == 0).astype(np.uint8)*255)
    cv2.imwrite('largest.png', bulb_mask*255)
    refined_mask[bulb_mask == 0] = 0

    bulb_mask = get_largest_component((refined_mask[:, :, 0] == 0).astype(np.uint8), 0)
    cv2.imwrite('bulb_mask.png', (refined_mask[:, :, 0] == 0).astype(np.uint8)*255)
    cv2.imwrite('largest.png', bulb_mask*255)
    refined_mask[bulb_mask == 0] = 0

    bg_mask = get_largest_component((refined_mask[:, :, 0] == 0).astype(np.uint8), 1)
    refined_mask[(bulb_mask == 0) & (bg_mask == 0)] = GrayscaleColors.IGNORE_INDEX_2.value
    refined_mask = cv2.medianBlur(refined_mask, 3)

    cv2.imwrite(os.path.join(refined_dir, mask_filename.replace('_mask', '')), refined_mask)

    ignore_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    red_pixels = np.array([0, 0, 255], dtype=np.uint8)
    
    if np.any(ignore_mask > 0):
        indices = np.where(ignore_mask > 0)  
        ignore_color[indices[0], indices[1]] = red_pixels

   