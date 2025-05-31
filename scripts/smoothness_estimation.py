import os
import sys
import argparse
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import tifffile
import csv
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from PIL import Image
from scipy.ndimage import convolve
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))
from file_utils import get_inference_dirs
from utils import get_model
from metrics import compute_metrics
from residual_unet import ResidualUnet
import importlib



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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation errors per epoch.")
    parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
    # parser.add_argument("--models_dir", required=True, help="Path to directory with model checkpoints")
    # parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment timestamp")
    # parser.add_argument("--num_slices", type=int, default=4, help="Number of slices per interval")
    # parser.add_argument("--epoch_step", type=int, default=10, help="Epoch step for model evaluation")
    parser.add_argument("--pred_dir", required=True, help="Path to predictions directory")
    # parser.add_argument("--output_csv", required=True, help="Path to output CSV file")
    return parser.parse_args()

def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module

def extract_number(filename):
    number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(number) if number else 0

def select_middle_slices(filenames, num_slices=4):
    selected = []
    for filename in filenames:
        slice_idx = extract_number(filename)
        res =  slice_idx % 10 
        if res in [4, 5, 6, 7]:
            selected.append(filename)
    return selected


def count_misclassified_pixels(pred, mask):
    valid_mask = mask != -1
    # print(valid_mask.shape)
    cv2.imwrite('valid.png', (valid_mask.squeeze().cpu().numpy()).astype(np.uint8)*255)
    mistakes = torch.zeros_like(mask)
    mistakes[(pred != mask) & valid_mask] = 1
    # print('mistake', mistakes.sum().item())
    return mistakes.sum().item()


def compute_iou(preds, targets, device, config):
    tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, ignore_index=config.SegmentationClass.IGNORE_INDEX.value, num_classes=config.NUM_CLASSES, mode=config.LOSS_MODE)
    # print(f"preds shape: {preds.shape}, targets shape: {targets.shape}")
    # print(f"preds: {torch.unique(preds)}, targets: {torch.unique(targets)}")

    # print("tp, fp, fn, tn", tp, fp, fn, tn)
    
    return smp.metrics.iou_score(tp.to(device), fp.to(device), fn.to(device), tn.to(device), 'micro-imagewise').item()

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

def create_color_map():
    # colors = [(0, 0, 0), (128, 0, 0), (255, 0, 0), (255, 127, 0), (255, 255, 0), (255, 255, 255)]
    colors = [(0, 0, 0), (255, 200, 5), (255, 150, 5), (255, 100, 5), (255, 50, 5), (255, 20, 5), (255, 0, 5)]
    color_map = np.array(colors, dtype=np.uint8)
    return color_map

def apply_custom_colormap(image, color_map):
    indexed_image = np.take(color_map, image, axis=0)
    return indexed_image

def show_mistakes(mask, pred, path, config):
    # print(np.unique(mask))
    # predicted_classes = get_class_matrix(predicted, config)
    # mask_classes = get_class_matrix(mask, config)

    predicted_classes = pred.squeeze().cpu().numpy()
    mask_classes = mask.squeeze().cpu().numpy()

    # print('predicted_classes shape', predicted_classes.shape)
    # print('mask_classes shape', mask_classes.shape)

    # print('classes', np.unique(predicted_classes), np.unique(mask_classes))

    predicted_classes[mask_classes == -1] = 0
    mask_classes[mask_classes == -1] = 0

    diff = np.abs(predicted_classes - mask_classes)
    cv2.imwrite('diff.png', diff*255)
    
    diff[mask_classes == -1] = 0
    # print(np.unique(diff))
    diff = diff.astype(np.int64)


    color_map = create_color_map()
    mistakes = apply_custom_colormap(diff, color_map)
    # print('diff', np.unique(diff))
    # print(np.unique(mistakes))

    mistakes = Image.fromarray(mistakes)
    mistakes.save(path)

    # cv2.imwrite(path, mistakes)
     
# def compute_change_map(volume, filenames):
#     depth = volume.shape[0]
#     change_map = np.zeros_like(volume, dtype=np.uint8)

#     for i in range(1, depth):
#         change_map[i-1] = (volume[i] != volume[i-1]).astype(np.uint8)

#     # change_map = np.concatenate([np.zeros_like(volume[-1:], dtype=np.uint8), change_map], axis=0)

#     for i in range(0, depth):
#         name = f'change_map_{filenames[i].replace('.png', '')}.png'
#         # {filenames[i].replace('.png', '')}
#         save_path = os.path.join('change_maps', name)
#         cv2.imwrite(save_path, change_map[i-1] * 255)

#     return change_map

def compute_change_map(volume, filenames):
    depth = volume.shape[0]
    change_map = np.zeros_like(volume, dtype=np.uint8)

    for i in range(1, depth):
        diff = (volume[i] != volume[i-1])
        ignore_mask = (volume[i] == -1) | (volume[i-1] == -1) 
        diff[ignore_mask] = 0  
        change_map[i] = diff.astype(np.uint8)


    return change_map

def compute_unstable_mask(change_map, neighborhood=1, threshold=1):
    kernel_size = 2 * neighborhood + 1
    kernel = np.ones((kernel_size, 1, 1), dtype=np.uint8)
    local_sum = convolve(change_map, kernel, mode='constant', cval=0)
    return (local_sum > threshold).astype(np.uint8)

def main():
    args = parse_args()
    config = load_config(args.config)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_dir = 'instability'
    change_map_dir = 'change_maps'

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(change_map_dir, exist_ok=True)

    filenames = sorted(os.listdir(args.pred_dir), key=extract_number)
    preds = []

    for filename in filenames:
        pred_path = os.path.join(args.pred_dir, filename)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred = preprocess_mask(pred, config) 
        preds.append(pred)

    preds = np.array(preds)

    change_map = compute_change_map(preds, filenames)

    for i in range(0, len(filenames)):
        # print(filenames[i])
        save_path = os.path.join(change_map_dir, f'{filenames[i]}')
        # save_path = os.path.join(save_dir, f'unstable_{i-1}.png')
        cv2.imwrite(save_path, change_map[i]*255)

    instability_mask = compute_unstable_mask(change_map)
    print(change_map.shape)
    print(instability_mask.shape)

    for i in range(0, len(filenames)):
        # print(filenames[i])
        save_path = os.path.join(save_dir, f'{filenames[i]}')
        # save_path = os.path.join(save_dir, f'unstable_{i-1}.png')
        cv2.imwrite(save_path, instability_mask[i]*255)

        
if __name__ == "__main__":
    main()
