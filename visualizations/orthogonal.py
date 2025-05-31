import os
import sys
from pathlib import Path
import logging
import torch
import cv2
import numpy as np
import tifffile
import argparse
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))


def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module

def setup_experiment_dirs(config, timestamp):
    experiment_dir = Path(config.EXPERIMENTS_DIR) / timestamp
    log_dir = experiment_dir / 'logs'
    metrics_dir = experiment_dir / 'metrics'
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir, log_dir, metrics_dir

parser = argparse.ArgumentParser(description='Process segmentation paths.')
parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
parser.add_argument("--timestamp", required=True, help="Experiment timestamp")
parser.add_argument('--img_dir', type=str, default=None, help='Path to image directory')
# parser.add_argument('--mask_dir', type=str, required=True, help='Path to mask directory')
# parser.add_argument('--pred_dir', type=str, required=True, help='Path to predictions directory')
# parser.add_argument('--save_dir', type=str, required=True, help='Directory to save')
parser.add_argument("--epoch", required=True, help="Experiment epoch")
parser.add_argument('--save_slices', action='store_true', help='Flag to save slices')
parser.add_argument('--save_color', action='store_true', help='Flag to save color slices')

args = parser.parse_args()
config = load_config(args.config)

experiment_dir, _, metrics_dir = setup_experiment_dirs(config, args.timestamp)

output_csv = os.path.join(metrics_dir, f"{args.timestamp}_boundary_mistakes.csv")
save_dir = os.path.join(experiment_dir, f"{args.timestamp}_{args.epoch}")
mask_dir = os.path.join(save_dir, "masks")
pred_dir = os.path.join(save_dir, "predictions")
# img_dir = args.img_dir #os.path.join(save_dir, "slices")

print(mask_dir, pred_dir)


zx_mask_dir = f"{save_dir}/zx"
zy_mask_dir = f"{save_dir}/zy"

os.makedirs(zx_mask_dir, exist_ok=True)
os.makedirs(zy_mask_dir, exist_ok=True)

print('Creating orthogonal visualizations...')

if args.save_slices:
    zx_img_dir = f"{save_dir}/zx_slices"
    zy_img_dir = f"{save_dir}/zy_slices"
    os.makedirs(zx_img_dir, exist_ok=True)
    os.makedirs(zy_img_dir, exist_ok=True)
else:
    zx_img_dir, zy_img_dir = None, None

if args.save_color:
    color_masks_dir = f"{save_dir}/color_masks_epoch"
    color_preds_dir = f"{save_dir}/color_preds_epoch"
    os.makedirs(color_masks_dir, exist_ok=True)
    os.makedirs(color_preds_dir, exist_ok=True)
else:
    color_masks_dir, color_preds_dir = None, None

volume_img = []
volume_masks = []
volume_preds = []
filenames = []

stripe_positions = [100, 120]
colors = [(0, 255, 0), (255, 0, 0)]

def load_mask(file_path):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask = np.where(mask == 255, 235, mask)  # Change 255 to 235 for clarity
    return mask

def load_image(file_path):
    image = tifffile.imread(file_path)
    return image

def extract_number(filename):
    basename = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, basename))
    return int(number)

def apply_color_map(mask, colormap=cv2.COLORMAP_JET):
    color_mask = cv2.applyColorMap(mask.astype(np.uint8), colormap)
    return color_mask

def change_color(mask, init_color=255, new_color=225):
    mask = np.where(mask == init_color, new_color, mask)
    return mask

def save_color_masks(masks, save_dir):
    for i, mask in enumerate(masks):
        color_mask = apply_color_map(mask)
        cv2.imwrite(os.path.join(save_dir, f'reco_{str(i+1).zfill(4)}.png'), color_mask)

def add_horizontal_stripes(data, stripe_positions, colors, thickness=1, alpha=0.5):
    striped_data = data.copy()
    overlay = striped_data.copy()

    height, width = striped_data.shape[:2]

    for position, color in zip(stripe_positions, colors):
        position = min(position, height - 1)
        cv2.line(overlay, (0, position), (width - 1, position), color, thickness=thickness)

    cv2.addWeighted(overlay, alpha, striped_data, 1 - alpha, 0, striped_data)

    return striped_data

# Process each image
for mask_filename in sorted(os.listdir(mask_dir), key=extract_number):
    # img_path = os.path.join(img_dir, mask_filename.replace('.png', '.tif'))
    mask_path = os.path.join(mask_dir, mask_filename)
    pred_path = os.path.join(pred_dir, mask_filename)

    if not os.path.exists(mask_path):
        continue

    filename = os.path.splitext(os.path.basename(mask_filename))[0]

    # img = load_image(img_path)
    mask = load_mask(mask_path)

    pred = load_mask(pred_path)
    # pred = cv2.resize(pred, dsize=(1616, 1152), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
 
    # volume_img.append(img)
    volume_masks.append(mask)
    volume_preds.append(pred)
    filenames.append(filename)

# volume_img = np.array(volume_img)
volume_masks = np.array(volume_masks)
volume_preds = np.array(volume_preds)

if color_masks_dir and color_preds_dir:
    save_color_masks(volume_masks, color_masks_dir)
    save_color_masks(volume_preds, color_preds_dir)

# Process ZX axis
if len(volume_masks.shape) == 3:
    # zx_img = np.transpose(volume_img, (2, 0, 1))  # (H, W, N)
    zx_mask = np.transpose(volume_masks, (2, 0, 1))  # (H, W, N)
    zx_pred = np.transpose(volume_preds, (2, 0, 1))  # (H, W, N)
else:
    raise ValueError(f"Unexpected volume_img shape: {volume_img.shape}")

os.makedirs(f'orthogonal_gt', exist_ok=True)

for i in range(zx_mask.shape[0]):
    # if zx_img_dir:
    #     slice_path = os.path.join(zx_img_dir, f"orthogonal_zx_{str(i).zfill(4)}.tif")
    #     tifffile.imwrite(slice_path, zx_img[i, :, :])

    mask_path = os.path.join(zx_mask_dir, f"orthogonal_zx_{str(i).zfill(4)}.png")
    color_mask = apply_color_map(zx_mask[i, :, :])
    gray_pred = np.uint8(zx_pred[i, :, :])

    vis = cv2.addWeighted(color_mask, 0.25, cv2.cvtColor(gray_pred, cv2.COLOR_GRAY2BGR), 1.0, 0.0)
    # vis = add_horizontal_stripes(vis, stripe_positions, colors)
    cv2.imwrite(mask_path, gray_pred) #TODO: restore cv2.imwrite(mask_path, vis)
    print(f'ZX-visualizations saved to {mask_path}')

    cv2.imwrite(f"orthogonal_gt/reco_{str(i).zfill(4)}.png", zx_mask[i, :, :])


# Process ZY axis
if len(volume_masks.shape) == 3:
    # zy_img = np.transpose(volume_img, (1, 0, 2))  # (H, W, N)
    zy_mask = np.transpose(volume_masks, (1, 0, 2))  # (H, W, N)
    zy_pred = np.transpose(volume_preds, (1, 0, 2))  # (H, W, N)
else:
    raise ValueError(f"Unexpected volume_img shape: {volume_img.shape}")

for i in range(zy_mask.shape[0]):
    # if zy_img_dir:
    #     slice_path = os.path.join(zy_img_dir, f"orthogonal_zy_{str(i).zfill(4)}.tif")
    #     tifffile.imwrite(slice_path, zy_img[i, :, :])

    mask_path = os.path.join(zy_mask_dir, f"orthogonal_zy_{str(i).zfill(4)}.png")
    color_mask = apply_color_map(zy_mask[i, :, :])
    gray_pred = np.uint8(zy_pred[i, :, :])

    vis = cv2.addWeighted(color_mask, 0.25, cv2.cvtColor(gray_pred, cv2.COLOR_GRAY2BGR), 1.0, 0.0)
    vis = add_horizontal_stripes(vis, stripe_positions, colors)
    cv2.imwrite(mask_path, gray_pred) #TODO: restore cv2.imwrite(mask_path, vis)
    print(f'ZY-visualizations saved to {mask_path}')

logging.info("Processing complete!")
