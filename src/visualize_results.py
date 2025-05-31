import os
import numpy as np
import cv2
from PIL import Image
from src.file_utils import ensure_directory_exists


def tensor_to_np(tensor):
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: Converted NumPy array.
    """

    np_array = tensor.cpu().numpy()
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3]:
        np_array = np_array.transpose(1, 2, 0)
    return np_array


def get_color_masks(mask):
    colors = [(0, 255, 0),     
              (0, 0, 255),    
              (255, 255, 0),   
              (128, 0, 128),    
              (0, 255, 255)]    

    classes = [69, 109, 153, 204, 255]
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


def load_grayscale_images(original_path, mask_path, predicted_path, logger):
    """Load grayscale images from specified paths.

    Args:
        original_path (str): Path to the original image.
        mask_path (str): Path to the mask image.
        predicted_path (str): Path to the predicted image.
        logger (logging.Logger): Logger for logging errors.

    Returns:
        tuple: NumPy arrays of the original, mask, and predicted images.

    Raises:
        Exception: If any image fails to load.
    """

    try:
        original = Image.open(original_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        predicted = Image.open(predicted_path).convert('L')
        return np.array(original).astype(np.uint8), np.array(mask).astype(np.uint8), np.array(predicted).astype(np.uint8)
    except Exception as e:
        logger.error(f"Failed to load images from {original_path}, {mask_path}, {predicted_path}. Exception: {e}")
        raise


def alpha_blend_images(images, alphas, logger):
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
    return edge_img


def save_image(output_path, filename, image_np, logger):
    """Save a NumPy image to a specified path.

    Args:
        output_path (str): Directory path to save the image.
        filename (str): Filename for the saved image.
        image_np (np.ndarray): Image to save.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If the image fails to save.
    """

    ensure_directory_exists(output_path)
    full_path = os.path.join(output_path, filename)
    try:
        image_pil = Image.fromarray(image_np)
        image_pil.save(full_path)
    except Exception as e:
        logger.error(f"Failed to save image to {full_path}. Exception: {e}")


def overlay_predictions(original, mask, predicted, output_dir, filename_base, config, logger):
    """Overlay predictions on the original image and save the result.

    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Ground truth mask.
        predicted (np.ndarray): Predicted mask.
        output_dir (str): Output directory for the overlay image.
        filename_base (str): Base filename for the saved image.
        config: Configuration object containing overlay parameters.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If overlay creation or saving fails.
    """

    try:
        overlayed = alpha_blend_images([original, mask, predicted], config.PREDICTION_ALPHAS, logger)
        save_image(os.path.join(output_dir, config.PREDICTION_OVERLAY_DIR), f"{filename_base}.png", overlayed, logger)
    except Exception as e:
        logger.error(f"Failed to create and save overlays for {filename_base}. Exception: {e}")


def overlay_colored_predictions(original, mask, predicted, output_dir, filename_base, config, logger):
    """Create and save an overlay of colored predictions on the original image.

    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Ground truth mask.
        predicted (np.ndarray): Predicted mask.
        output_dir (str): Output directory for the colored overlay image.
        filename_base (str): Base filename for the saved image.
        config: Configuration object containing overlay parameters.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If overlay creation or saving fails.
    """

    try:
        original_color_np = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0] = config.TARGETS_COLOR

        colored_pred = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)
        colored_pred[predicted > 0] = config.PREDICTIONS_COLOR

        colored_overlayed = alpha_blend_images([original_color_np, colored_mask, colored_pred], config.PREDICTION_ALPHAS, logger)
        save_image(os.path.join(output_dir, config.COLORED_PREDICTION_OVERLAY_DIR), f"{filename_base}.png", colored_overlayed, logger)
    except Exception as e:
        logger.error(f"Failed to create and save overlays for {filename_base}. Exception: {e}")


def overlay_true_masks(original, mask, predicted, output_dir, filename_base, config, logger):
    """Overlay true masks on the original image and save the result.

    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Ground truth mask.
        output_dir (str): Output directory for the mask overlay image.
        filename_base (str): Base filename for the saved image.
        config: Configuration object containing overlay parameters.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If overlay creation or saving fails.
    """

    try:
        blended_mask = alpha_blend_images([original, mask], config.MASK_ALPHAS, logger)
        save_image(os.path.join(output_dir, config.MASK_OVERLAY_DIR), f"{filename_base}.png", blended_mask, logger)
    except Exception as e:
        logger.error(f"Failed to overlay true masks for {filename_base}. Exception: {e}")


def overlay_edges(original, mask, predicted, output_dir, filename_base, config, logger):
    """Overlay edges of masks and predictions on the original image and save the result.

    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Ground truth mask.
        predicted (np.ndarray): Predicted mask.
        output_dir (str): Output directory for the edge overlay image.
        filename_base (str): Base filename for the saved image.
        config: Configuration object containing overlay parameters.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If edge overlay creation or saving fails.
    """

    try:
        edge_mask = apply_morphological_edge_detection(mask)
        edge_pred = apply_morphological_edge_detection(predicted)
        edge_overlayed = alpha_blend_images([original, edge_mask, edge_pred], config.EDGE_ALPHAS, logger)
        save_image(os.path.join(output_dir, config.EDGE_OVERLAY_DIR), f"{filename_base}_overlayed_edges.png", edge_overlayed, logger)
    except Exception as e:
        logger.error(f"Failed to create and save edges overlay for {filename_base}. Exception: {e}")


def overlay_colored_edges(original, mask, predicted, output_dir, filename_base, config, logger):
    """Overlay colored edges of masks and predictions on the original image and save the result.

    Args:
        original (np.ndarray): Original image.
        mask (np.ndarray): Ground truth mask.
        predicted (np.ndarray): Predicted mask.
        output_dir (str): Output directory for the colored edge overlay image.
        filename_base (str): Base filename for the saved image.
        config: Configuration object containing overlay parameters.
        logger (logging.Logger): Logger for logging errors.

    Raises:
        Exception: If colored edge overlay creation or saving fails.
    """
    
    try:
        edge_mask = apply_morphological_edge_detection(mask)
        edge_pred = apply_morphological_edge_detection(predicted)

        ignore_mask = (mask == 127).astype(np.uint8) * 255
        cv2.imwrite("ignore_mask.png", ignore_mask)

        original_color_np = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

        ignore_mask_bgr = cv2.cvtColor(ignore_mask, cv2.COLOR_GRAY2BGR)

        colored_edge_mask = np.zeros((edge_mask.shape[0], edge_mask.shape[1], 3), dtype=np.uint8)
        colored_edge_pred = np.zeros((edge_pred.shape[0], edge_pred.shape[1], 3), dtype=np.uint8)
        
        colored_edge_mask[edge_mask > 0] = config.TARGETS_COLOR
        colored_edge_pred[edge_pred > 0] = config.PREDICTIONS_COLOR

        colored_edge_overlayed = alpha_blend_images([original_color_np, colored_edge_mask, colored_edge_pred, ignore_mask_bgr], config.EDGE_ALPHAS, logger)

        save_image(os.path.join(output_dir, config.COLORED_EDGE_OVERLAY_DIR), f"{filename_base}.png", colored_edge_overlayed, logger)

    except Exception as e:
        logger.error(f"Failed to create and save edges overlay for {filename_base}. Exception: {e}")


# def overlay_mask_edges(original, mask, predicted, output_dir, filename_base, config, logger):
#     try:
#         edge_mask = apply_morphological_edge_detection(mask)
#         edge_overlayed = alpha_blend_images([original, edge_mask], config.MASK_EDGE_ALPHAS, logger)
#         save_image(os.path.join(output_dir, config.MASK_EDGE_OVERLAY_DIR), f"{filename_base}_overlayed_edges.png", edge_overlayed, logger)
#     except Exception as e:
#         logger.error(f"Failed to overlay true edges for {filename_base}. Exception: {e}")

def overlay_mask_edges(original, mask, predicted, output_dir, filename_base, config, logger):
    try:
        # edge_mask = apply_morphological_edge_detection(mask)
        edge_mask =  get_color_masks(mask)

        ignore_mask = ((mask == 127) | (mask == 50)).astype(np.uint8) * 255
        ignore_mask = cv2.cvtColor(ignore_mask, cv2.COLOR_GRAY2BGR)  

        if len(original.shape) == 2:  
            original_color_np = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)  
        else:
            original_color_np = original  
        colored_edge_overlayed = alpha_blend_images([original_color_np, edge_mask, ignore_mask], [1.0, 0.6, 0.3], logger)
        save_image(os.path.join(output_dir, config.MASK_EDGE_OVERLAY_DIR), f"{filename_base}.png", colored_edge_overlayed, logger)

    except Exception as e:
        print(f"Failed to create and save edges overlay for {filename_base}. Exception: {e}")


def get_class_matrix(matrix, config):    
    mask = np.zeros(matrix.shape, np.float64)
    
    conditions = [
        matrix == 69,
        matrix == 109,
        matrix == 153,
        matrix == 204,
        matrix == 255, 
        matrix == 127
    ]
    
    values = [1, 2, 3, 4, 5, -1]
    
    for condition, value in zip(conditions, values):
        mask[condition] = value
    
    # print(np.unique(mask))
    return mask

def create_color_map():
    # colors = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 127, 0), (255, 255, 0), (255, 255, 255)]
    colors = [(0, 0, 0), (255, 200, 5), (255, 150, 5), (255, 100, 5), (255, 50, 5), (255, 20, 5), (255, 0, 5)]
    color_map = np.array(colors, dtype=np.uint8)
    return color_map

def apply_custom_colormap(image, color_map):
    indexed_image = np.take(color_map, image, axis=0)
    return indexed_image

def show_mistakes(original, mask, predicted, output_dir, filename_base, config, logger):
    predicted_classes = get_class_matrix(predicted, config)
    mask_classes = get_class_matrix(mask, config)

    predicted_classes[mask_classes == -1] = 0
    mask_classes[mask_classes == -1] = 0

    diff = np.abs(predicted_classes - mask_classes)
    cv2.imwrite('diff.png', diff*255)
    
    diff[mask_classes == -1] = 0
    diff = diff.astype(np.int64)

    os.makedirs(os.path.join(output_dir, config.MISTAKES_DIR), exist_ok=True)

    color_map = create_color_map()
    mistakes = apply_custom_colormap(diff, color_map)

    save_image(os.path.join(output_dir, config.MISTAKES_DIR), f"{filename_base}.png", mistakes, logger)
     

def visualize(original_paths, masks_paths, predicted_paths, output_dir, config, logger):
    ensure_directory_exists(output_dir)
    for i in range(len(original_paths)):
    # for i, (image, mask, filename) in enumerate(loader):
        try:
            original, mask, predicted = load_grayscale_images(original_paths[i], masks_paths[i], predicted_paths[i], logger)
            base_name = os.path.basename(original_paths[i])
            base_name_no_suffix = base_name.replace('_original', '')
            filename_base, _ = os.path.splitext(base_name_no_suffix)

            # filename = filename[0] if isinstance(filename, tuple) else filename
            # filename_base, _ = os.path.splitext(filename)
            # image = image.squeeze().squeeze().cpu().numpy()
            # mask = mask.squeeze().squeeze().cpu().numpy()
            # predicted = predictions[i]

            # print('mask', np.unique(mask), mask.shape)
            # print('pred', np.unique(predicted), predicted.shape)

            visualizations = [
                (overlay_predictions, config.ENABLE_OVERLAY_PREDICTIONS),
                (overlay_colored_predictions, config.ENABLE_OVERLAY_COLORED_PREDICTIONS),
                (overlay_true_masks, config.ENABLE_OVERLAY_TRUE_MASKS),
                (overlay_edges, config.ENABLE_OVERLAY_EDGES),
                (overlay_colored_edges, config.ENABLE_OVERLAY_COLORED_EDGES),
                (overlay_mask_edges, config.ENABLE_OVERLAY_MASK_EDGES),
                (show_mistakes, config.ENABLE_SHOW_MISTAKES),
            ]

            for func, is_enabled in visualizations:
                if is_enabled:
                    func(original, mask, predicted, output_dir, filename_base, config, logger)

        except Exception as e:
            logger.error(f"Failed to visualize images. Exception: {e}")
