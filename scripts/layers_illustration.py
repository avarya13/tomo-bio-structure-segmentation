import os
import numpy as np
import cv2
import tifffile


def alpha_blend_images(images, alphas):
    """Blend multiple images using specified alpha coefficients."""
    if len(images) != len(alphas):
        raise ValueError("The number of images and alpha coefficients must be the same.")

    blended_img = np.zeros_like(images[0], dtype=np.float32)
    for img, alpha in zip(images, alphas):
        blended_img += alpha * img
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    return blended_img


def apply_morphological_edge_detection(image):
    """Apply morphological edge detection to an image."""
    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    edge_img = dilated_img - image
    return edge_img


image_path = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data_layers_seq/test/x/reco_5.tif'
mask_path = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data_layers_seq/test/y_/reco_5.png'
save_path = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/reco_5_overlay.png'

image = tifffile.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  

edge_mask = apply_morphological_edge_detection(mask)

colored_edge_mask = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 3), dtype=np.uint8)
colored_edge_mask[edge_mask > 0] = [0, 255, 0]

image = (image * 255).astype(np.uint8)

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

colored_edge_overlayed = alpha_blend_images([image, colored_edge_mask], [1.0, 0.85])

os.makedirs(os.path.dirname(save_path), exist_ok=True)
cv2.imwrite(save_path, colored_edge_overlayed)

print(f"Сохранено: {save_path}")
