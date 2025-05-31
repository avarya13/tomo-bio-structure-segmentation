import os
import sys
import torch
from PIL import Image
import numpy as np
from file_utils import ensure_directory_exists

class_to_intensity = {
    # 0: 0,    # BACKGROUND
    # 2: 69,   # GL
    # 1: 153,  # SL (surface)
    # 3: 109,  # EPL
    # 4: 255,  # MCL
    # 5: 204,   # GCL
    # -1: 127,
    # # -1: 50
    0: 0,    # BACKGROUND
    2: 109,   # GL
    1: 69,  # SL (surface)
    3: 153,  # EPL
    4: 204,  # MCL
    5: 255,   # GCL
    -1: 127,
    # -1: 50
}


def tensor_to_np(tensor):
    np_array = tensor.cpu().numpy()
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3]:
        np_array = np_array.transpose(1, 2, 0)
    return np_array


""" def count_pixels(image, filename, config, logger, isTarget):
    pixels_type = 'target' if isTarget else 'predicted'
    logger.info(f'Total count of {pixels_type} pixels for the image {filename}: {image.shape[0]*image.shape[1]}')
    for cls in config.SegmentationClass:
        count = (image == cls.value).sum()        
        logger.info(f'Count of {pixels_type} pixels of class {cls.name} for the image {filename}: {count}') """


def predict(model, test_loader, device, config, logger, results_dir=None):
    """
    Perform inference on a test dataset using the specified model.

    This function evaluates the model on the provided test DataLoader, 
    computes predictions, and saves the predicted masks and ground truth 
    masks as images. It also logs pixel counts for each class if needed.

    Args:
        model (torch.nn.Module): The model used for making predictions.
        test_loader (DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device to run the model on (CPU or GPU).
        config (Config): Configuration object containing settings and class definitions.
        logger (logging.Logger): Logger object for logging information and errors.
        results_dir (str, optional): Directory to save prediction and mask images. 
                                      If None, results are not saved. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Tensor: Concatenated ground truth masks for all images.
            - Tensor: Concatenated predicted masks for all images.
    """
    
    model.eval()
    masks_list = []
    predicted_list = []

    for inputs, masks, filenames, _ in test_loader:
        inputs, masks = inputs.to(device), masks.to(device)

        masks = masks.squeeze(1).long() # for CE

        # print(torch.unique(masks))

        with torch.no_grad():
            preds = model(inputs)

        if config.LOSS_MODE == 'binary':
            preds = (torch.sigmoid(preds) > 0.5).long()
        elif config.LOSS_MODE == 'multiclass':
            preds = torch.argmax(preds, dim=1).long()

        masks = masks.long().cpu()
        preds = preds.cpu()

        masks_list.append(masks)  
        predicted_list.append(preds)

        """ for cls in config.SegmentationClass:
            count = (masks == cls.value).sum()
            print(torch.unique(masks))
            #pixels_type = 'target'
            logger.info(f'Count of target pixels of class {cls.name}: {count}')

        for cls in config.SegmentationClass:
            count = (preds == cls.value).sum()
            print(torch.unique(preds))
            #pixels_type = 'predicted'
            logger.info(f'Count of predicted pixels of class {cls.name}: {count}') """

        preds_np = preds.numpy() #.astype(np.uint8)
        masks_np = masks.numpy() #.astype(np.uint8)
       

        if results_dir:
            preds_dir = os.path.join(results_dir, 'predictions')
            masks_dir = os.path.join(results_dir, 'masks')
            originals_dir = os.path.join(results_dir, 'slices')

            ensure_directory_exists(preds_dir)
            ensure_directory_exists(masks_dir)
            ensure_directory_exists(originals_dir)

            for i in range(len(preds_np)):
                # pred_img = preds_np[i][0]
                # mask_img = masks_np[i][0]

                # CE

                pred_img = preds_np[i]
                mask_img = masks_np[i]

                original_img = tensor_to_np(inputs[i])

                """ count_pixels(mask_img, filenames[i], config, logger, True)
                count_pixels(pred_img, filenames[i], config, logger, False) """
                
                if original_img.max() <= 1.0:
                    original_img = (original_img * 255).astype(np.uint8)
                
                if original_img.ndim == 3 and original_img.shape[2] == 1:
                    original_img = original_img[:, :, 0]
                elif original_img.ndim == 3 and original_img.shape[2] == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported image shape: {original_img.shape}")

                # for multiclass

                pred_img = np.vectorize(class_to_intensity.get)(pred_img)
                mask_img = np.vectorize(class_to_intensity.get)(mask_img)

                pred_img_pil = Image.fromarray(pred_img.astype(np.uint8), mode='L')
                mask_img_pil = Image.fromarray(mask_img.astype(np.uint8), mode='L')
                original_img_pil = Image.fromarray(original_img)

                # pred_img_pil = Image.fromarray(pred_img * 255, mode='L')
                # mask_img_pil = Image.fromarray(mask_img * 255, mode='L')
                # original_img_pil = Image.fromarray(original_img)

                pred_img_pil.save(os.path.join(preds_dir, f"{os.path.splitext(filenames[i])[0]}.png"))
                mask_img_pil.save(os.path.join(masks_dir, f"{os.path.splitext(filenames[i])[0]}.png"))
                original_img_pil.save(os.path.join(originals_dir, f"{os.path.splitext(filenames[i])[0]}.png"))

    return torch.cat(masks_list, dim=0).to(device), torch.cat(predicted_list, dim=0).to(device)
