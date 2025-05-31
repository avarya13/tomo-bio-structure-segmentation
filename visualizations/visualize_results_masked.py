import os
import sys
from pathlib import Path
import importlib
import argparse
import numpy as np
import tifffile
import cv2
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

def setup_experiment_dirs(config, timestamp):
    experiment_dir = Path(config.EXPERIMENTS_DIR) / timestamp
    log_dir = experiment_dir / 'logs'
    metrics_dir = experiment_dir / 'metrics'
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir, log_dir, metrics_dir

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

def show_mistakes(mask, predicted, weights, output_dir, filename_base, config):
    predicted_classes = get_class_matrix(predicted, config)
    mask_classes = get_class_matrix(mask, config)

    # predicted_classes[(mask_classes == -1) | (weights < 0.5)] = 0
    # mask_classes[(mask_classes == -1) | (weights < 0.5)] = 0

    weights_mask =  (weights > 0.5).astype(np.uint8)

    predicted_classes[(mask_classes == -1)] = 0
    mask_classes[(mask_classes == -1)] = 0

    predicted_classes *= weights_mask 
    mask_classes *= weights_mask 

    diff = np.abs(predicted_classes - mask_classes)

    cv2.imwrite('diff.png', diff*255)
    
    diff[mask_classes == -1] = 0
    diff = diff.astype(np.int64)

    color_map = create_color_map()
    mistakes = apply_custom_colormap(diff, color_map)

    mistakes = Image.fromarray(mistakes)
    mistakes.save(os.path.join(output_dir, f"{filename_base}.png"))

    print(f"Saved to: {os.path.join(output_dir, f"{filename_base}.png")}")



def main(args):
    config = importlib.import_module(args.config)
    experiment_dir, _, metrics_dir = setup_experiment_dirs(config, args.timestamp)
    save_dir = experiment_dir / f"{args.timestamp}_{args.epoch}/mistakes_weighted"

    mask_path = experiment_dir / f"{args.timestamp}_{args.epoch}/masks"
    pred_path = experiment_dir / f"{args.timestamp}_{args.epoch}/predictions"
    weights = tifffile.imread(config.LOSS_WEIGHTS_PATH)

    os.makedirs(save_dir, exist_ok=True)

    for i, (mask_file, pred_file) in enumerate(zip(os.listdir(mask_path), os.listdir(pred_path))):
        mask = np.array(Image.open(os.path.join(mask_path, mask_file)).convert('L')).astype(np.uint8)
        pred =  np.array(Image.open(os.path.join(pred_path, pred_file)).convert('L')).astype(np.uint8)
        filename_base = os.path.splitext(mask_file)[0]
        show_mistakes(mask, pred, weights[i], save_dir, filename_base, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Olfactory bulb segmentation project')
    parser.add_argument('--timestamp', type=str, help='The label of the experiment in the format: YYYYMMDD-HHMM') 
    parser.add_argument('--epoch', type=str, default=None, help='Specific epoch to test. Optional for single_infer.') 
    # parser.add_argument('--save', type=str, choices=['train', 'test', 'val', 'all'], default=None, help='Mode of operation: test, validation, or all (for full_infer)')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file to use without the .py extension")

    args = parser.parse_args()
    main(args)