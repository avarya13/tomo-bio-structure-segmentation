import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile
from pathlib import Path
import cv2


#overlay_path = Path(r"D:\datasets\enference.colored_cor\enference.colored_cor\bu_ori_0689_overlayed_colored.png")
pred_path = Path(r'D:\diploma\unet-olfactory-bulb-segmentation\bu_ori_0690_pred.png')
roi_path = Path(r"D:\datasets\луковица\experiments_data\v2_fixed\300_300_roi_v2\bu\x\bu_ori_0690.tif")
mask_v1_path = Path(r'D:\datasets\луковица\experiments_data\300_300_roi\bu\y\bu_ori_0690.png')
mask_v2_path = Path(r"D:\datasets\луковица\experiments_data\v2_fixed\300_300_roi_v2\bu\y\bu_ori_0690.png")


def alpha_blend_images(images, alphas):
    if len(images) != len(alphas):
        raise ValueError("The number of images and alpha coefficients must be the same.")

    for i in range(len(images)):
        if len(images[i].shape) == 2:  
            images[i] = np.stack([images[i]] * 3, axis=-1)
    
    blended_img = np.zeros_like(images[0], dtype=np.float32)
    for img, alpha in zip(images, alphas):
        blended_img += alpha * img
    blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
    return blended_img


def get_color_mask(images, color, predicate):
    colored_mask = np.zeros((images.shape[0], images.shape[1], 3), dtype=np.uint8)
    colored_mask[predicate(images)] = color    
    return colored_mask


def load_image_with_pillow(image_path):
    """Load image using Pillow and convert it to NumPy array."""
    image = Image.open(image_path)
    return np.array(image)


def create_mask(img, channel):
    diff = img - channel

    marking = np.zeros(img.shape, dtype=np.uint8)
    marking[(diff > 0)] = 255

    """ kernel = np.ones((3, 3), np.uint8)
    morf_marking = cv2.dilate(marking, kernel)
    morf_marking = cv2.erode(morf_marking, kernel) """
    
    return marking

def apply_morphological_edge_detection(image):
    """Apply morphological edge detection to an image.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        np.ndarray: Image with edges detected.
    """

    kernel = np.ones((5, 5), np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    edge_img = dilated_img - image
    return edge_img

def main():
    roi = (tifffile.imread(str(roi_path)) * 255).astype(np.uint8)  
    pred = load_image_with_pillow(pred_path)
    mask_1 = load_image_with_pillow(mask_v1_path) 
    mask_2 = load_image_with_pillow(mask_v2_path)  

    #pred = create_mask(roi, overlayed[:, :, 2])

    """ tp = apply_morphological_edge_detection(get_color_mask(pred, [255, 0, 0]))  # Blue for prediction
    mask_color_1 = apply_morphological_edge_detection(get_color_mask(mask_1, [0, 0, 255]))  # Red for first mask
    mask_color_2 = apply_morphological_edge_detection(get_color_mask(mask_2, [0, 255, 0]))  # Green for second mask """

    """ tp = get_color_mask(pred, [200, 0, 0], lambda x: x > 0)  # Blue for first mask
    mask_color_1 = get_color_mask(mask_1, [0, 0, 200], lambda x: x > 0)          # Blue for 1st mask
    #tp_color = get_color_mask((pred & mask_1) | (pred & mask_2), [0, 255, 0], lambda x: x > 0)  # Blue for first mask
    mask_color_2 = get_color_mask(mask_2 - mask_1, [0, 0, 255], lambda x: x > 0) # Blue for 2nd mask """

    tp_color = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    mask_color_1 = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    mask_color_2 = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    mask_color_1_ = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    mask_color_2_ = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    fp_color = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    fn_color = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)

    tp_color[(pred > 0) & ((mask_1 > 0) | (mask_2 > 0))] = [14,252,20]
    fp_color[(pred > 0) & ((mask_1 == 0) & (mask_2 == 0))] = [230, 15, 250]
    fn_color[(pred == 0) & ((mask_1 > 0) | (mask_2 > 0))] = [242, 62, 7]
    mask_color_1[(mask_1 > 0)] = [11, 33, 229]
    mask_color_2[(mask_1 == 0) & (mask_2 > 0)] = [7, 131, 247] 

    mask_color_1_[(mask_1 > 0) & (pred == 0)] = [0, 30, 255]
    mask_color_2_[(mask_1 == 0) & (mask_2 > 0) & (pred == 0)] = [0,0,255] 
    
    # Blending images
    images= [Image.fromarray(alpha_blend_images([roi, mask_color_1], [0.85, 0.95]))]
    #images.append(Image.fromarray(alpha_blend_images([roi, mask_color_1, mask_color_2, tp], [1.0, 0.8, 0.8, 0.8])))
    images.append(Image.fromarray(alpha_blend_images([roi, fn_color, tp_color, fp_color], [0.85, 0.95, 0.95, 1.0])))
    images.append(Image.fromarray(alpha_blend_images([roi, mask_color_1, mask_color_2], [0.85, 0.95, 0.95])))

    # Create collage
    collage_width = roi.shape[1] * 3
    collage_height = roi.shape[0]
    collage = Image.new('RGB', (collage_width, collage_height), 'white')

    draw = ImageDraw.Draw(collage)
    font = ImageFont.truetype("arial.ttf", 48)

    labels = ['a', 'b', 'c']

    for i, (image, label) in enumerate(zip(images, labels)):
        collage.paste(image, (i * image.width, 0))
        #draw.rectangle([i * image.width, collage_height - 50, i * image.width + 65, collage_height], fill="white")
        draw.text((i * image.width + 20, collage_height - 60), label, fill="white", font=font)
        
    collage.save('markups_collage.png')


if __name__ == "__main__":
    main()
