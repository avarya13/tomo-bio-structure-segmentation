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
from pathlib import Path
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))
from file_utils import get_inference_dirs
from residual_unet import ResidualUnet
import importlib


#TODO: move to scripts

def preprocess_mask(mask, config):
    """
    Preprocess the mask by converting colors to class labels.

    Parameters:
        mask (numpy.ndarray): The mask to preprocess.

    Returns:
        numpy.ndarray: The processed mask with class labels.
    """

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

    return mask.astype(np.float64)  

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation errors per epoch.")
    parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
    parser.add_argument("--timestamp", required=True, help="Experiment timestamp")
    parser.add_argument("--epoch_step", type=int, default=10, help="Epoch step for model evaluation")
    return parser.parse_args()

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


def count_misclassified_pixels(pred, mask):
    valid_mask = mask != -1
    mistakes = torch.zeros_like(mask)
    mistakes[(pred != mask) & valid_mask] = 1
    return mistakes.sum().item()

def compute_iou(preds, targets, device, config):
    tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, ignore_index=config.SegmentationClass.IGNORE_INDEX.value, num_classes=config.NUM_CLASSES, mode=config.LOSS_MODE)
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

# def get_last_epoch(csv_path):
#     if os.path.exists(csv_path):
#         with open(csv_path, mode='r') as csvfile:
#             reader = csv.reader(csvfile)            
#             next(reader) # Skip header row
#             rows = list(reader)
#             if rows:                
#                 last_epoch = int(rows[-1][1]) # Get the last row's epoch
#                 return last_epoch
#     return 0  

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment_dir, log_dir, metrics_dir = setup_experiment_dirs(config, args.timestamp)
    csv_path = metrics_dir / f"{args.timestamp}_stats_mid.csv"
    img_csv_path = metrics_dir / f"{args.timestamp}_stats_mid_img.csv"

    # Get the last epoch from CSV if it exists, otherwise start from epoch 0
    # last_epoch = get_last_epoch(csv_path)
    # print(f"Resuming from epoch: {last_epoch}")

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH), interpolation=transforms.InterpolationMode.NEAREST)
    ])  

    results = []
    img_results = []
    modes = ['test', 'val', 'train']

    os.makedirs("pred", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("mistakes", exist_ok=True)

    print(f'Calculating average error and IoU for each epoch...')

    for mode in modes:
        imgs_dir, masks_dir, _ = get_inference_dirs(mode, config)
        imgs_filenames = sorted(os.listdir(imgs_dir), key=extract_number)
        masks_filenames = sorted(os.listdir(masks_dir), key=extract_number)
        selected_imgs = select_middle_slices(imgs_filenames)

        print("images:", selected_imgs)
        print("exp:", experiment_dir)
        
        checkpoint_files = sorted([f for f in os.listdir(experiment_dir) if f.endswith(".pth")], key=extract_number)

        selected_checkpoints = [f for f in checkpoint_files if extract_number(f) % args.epoch_step == 0]
        # selected_checkpoints = [f for f in checkpoint_files if extract_number(f) > last_epoch and extract_number(f) % args.epoch_step == 0]
        print("checkpoints", selected_checkpoints)
        for checkpoint in selected_checkpoints:
            epoch = os.path.splitext(os.path.basename(checkpoint))[0].split('_')[-1]
            model = load_model(os.path.join(experiment_dir, checkpoint), device, config)

            print(f"Epoch: {epoch}, checkpoint: {checkpoint}")

            total_mistakes = []
            total_iou = []
            preds, targets = [], []

            for filename in selected_imgs:
                img_path = os.path.join(imgs_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.tif', '.png'))
                img = tifffile.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = preprocess_mask(mask, config)

                img = transform(img).to(device)
                mask = transform(mask)

                mask = mask.to(device).long()

                pred = predict_mask(model, img, device)

                mistakes = count_misclassified_pixels(pred, mask)
                iou = compute_iou(pred, mask, device, config)
 
                total_mistakes.append(mistakes)
                preds.append(pred)
                targets.append(mask)

                img_results.append([epoch, filename, mistakes, iou])

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            mean_iou = compute_iou(preds, targets, device, config)
        
            print(mode, 'iou', mean_iou)

            results.append([mode, epoch, round(np.mean(total_mistakes)), round(mean_iou, 3)])

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # if last_epoch == 0:
        writer.writerow(["Mode", "Epoch", "Mean Misclassified Pixels", "Mean IoU"])  # Write header only once
        writer.writerows(results)

    with open(img_csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # if last_epoch == 0:
        writer.writerow(["Epoch", "Image", "Misclassified Pixels", "IoU"])  # Write header only once
        writer.writerows(img_results)

if __name__ == "__main__":
    main()