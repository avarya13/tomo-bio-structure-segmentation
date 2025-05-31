import os
import sys
import importlib
import argparse
import torch
from torchvision import transforms
import numpy as np
import cv2
import tifffile
import csv
from pathlib import Path
from scipy.ndimage import convolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))
from file_utils import get_inference_dirs
from residual_unet import ResidualUnet


def preprocess_mask(mask, config):
    """
    Preprocess the mask by converting colors to class labels.

    Parameters:
        mask (numpy.ndarray): The mask to preprocess.

    Returns:
        numpy.ndarray: The processed mask with class labels.
    """

    # print(np.unique(mask))
    mask =  mask.astype(np.float64)
    class_colors = config.GrayscaleColors
    labels = config.SegmentationClass
    
    mask = np.where(mask == class_colors.BACKGROUND_COLOR.value, labels.BACKGROUND.value, mask)
    mask = np.where(mask == class_colors.SL.value, labels.SL.value, mask)
    mask = np.where(mask == class_colors.GL.value, labels.GL.value, mask)
    mask = np.where(mask == class_colors.EPL.value, labels.EPL.value, mask)
    mask = np.where(mask == class_colors.MCL.value, labels.MCL.value, mask)
    mask = np.where(mask == class_colors.GCL.value, labels.GCL.value, mask)
    if class_colors.IGNORE_INDEX.value:
        mask = np.where(mask == class_colors.IGNORE_INDEX.value, labels.IGNORE_INDEX.value, mask)
    
    if class_colors.IGNORE_INDEX_2.value:
        mask = np.where(mask == class_colors.IGNORE_INDEX_2.value, labels.IGNORE_INDEX_2.value, mask)

    # print(np.unique(mask))

    return mask.astype(np.float64)  


# def parse_args():
#     parser = argparse.ArgumentParser(description="Analyze segmentation errors per epoch.")
#     parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
#     parser.add_argument("--mid", action='store_true', help="A flag to assess smoothness only for center slices from each 10")
#     parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment timestamp")
#     # parser.add_argument("--num_slices", type=int, default=4, help="Number of slices per interval")
#     parser.add_argument("--epoch_step", type=int, default=10, help="Epoch step for model evaluation")
#     return parser.parse_args()

def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module

def extract_number(filename):
    number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(number) if number else 0

def select_middle_slices(filenames):
    selected = []
    for filename in filenames:
        slice_idx = extract_number(filename)
        res =  slice_idx % 10 
        if res in [4, 5, 6, 7]:
            selected.append(filename)
    return selected

def load_model(checkpoint_path, device, config):
    model = ResidualUnet(config.INPUT_CHANNELS, config.NUM_CLASSES, config.CH_AGGR_TYPE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval()
    return model

def predict_mask(model, image, device):
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(image)
    return torch.argmax(pred, dim=1)

def setup_experiment_dirs(config, timestamp):
    experiment_dir = Path(config.EXPERIMENTS_DIR) / timestamp
    log_dir = experiment_dir / 'logs'
    metrics_dir = experiment_dir / 'metrics'
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir, log_dir, metrics_dir

def compute_change_map(volume, filenames):
    depth = volume.shape[0]
    change_map = np.zeros_like(volume, dtype=np.uint8)

    for i in range(1, depth):
        change_map[i-1] = (volume[i] != volume[i-1]).astype(np.uint8)

    for i in range(0, depth):
        name = f'change_map_{filenames[i].replace('.png', '')}.png'
        # {filenames[i].replace('.png', '')}
        save_path = os.path.join('change_maps', name)
        cv2.imwrite(save_path, change_map[i] * 255)

    return change_map

def compute_unstable_mask(change_map, neighborhood=1, threshold=1):
    kernel_size = 2 * neighborhood + 1
    kernel = np.ones((kernel_size, 1, 1), dtype=np.uint8)
    local_sum = convolve(change_map, kernel, mode='constant', cval=0)
    return (local_sum > threshold).astype(np.uint8)


def main():
    # args = parse_args()
    # config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masks_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data_all/y_'
    save_dir = r'/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/instability'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("change", exist_ok=True)

    # transform = transforms.Compose([
    #     transforms.ToTensor(), 
    #     transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH), interpolation=transforms.InterpolationMode.NEAREST)
    # ])  

  
    masks = []
    masks_filenames = []
       
    for i, filename in enumerate(sorted(os.listdir(masks_dir), key=extract_number)):  
        if i % 10 != 0:
            continue
        mask_path = os.path.join(masks_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = preprocess_mask(mask, config)
        # mask = transform(mask).to(device)  
        masks_filenames.append(filename)
        masks.append(mask)
    masks = np.array(masks)

    change_map = compute_change_map(masks, masks_filenames)
    print(change_map.shape, np.unique(change_map))

    for filename, msk in zip(masks_filenames, change_map):
        save_path = os.path.join("change", filename)
        cv2.imwrite(save_path, msk*255)

    instability_mask = compute_unstable_mask(change_map)

    print(instability_mask.shape, np.unique(instability_mask))

    
    for i, msk in enumerate(instability_mask):  
        for j in range(10): 
            index = i*10 + j + 1    
            save_path = os.path.join(save_dir, f"reco_{str(index).zfill(4)}.png")
            cv2.imwrite(save_path, msk*255)
            print(index)


if __name__ == "__main__":
    main()