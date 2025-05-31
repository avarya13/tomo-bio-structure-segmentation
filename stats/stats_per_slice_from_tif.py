import os
import sys
from pathlib import Path
import cv2
import numpy as np
import csv
import tifffile
import argparse
import importlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))


def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation errors.")
    # parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
    # parser.add_argument("--timestamp", required=True, help="Experiment timestamp")
    parser.add_argument("--pred_path", required=True, help="")
    parser.add_argument("--masks_dir", required=True, help="")
    parser.add_argument("--save_dir", required=True, help="")
    parser.add_argument("--debug_dir", default=None, help="Path to save debug images (optional)")
    return parser.parse_args()

def setup_experiment_dirs(config, timestamp):
    experiment_dir = Path(config.EXPERIMENTS_DIR) / timestamp 
    log_dir = experiment_dir / 'logs'
    metrics_dir = experiment_dir / 'metrics'
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    return experiment_dir, log_dir, metrics_dir

def extract_number(filename):
    basename = os.path.basename(filename)
    number = ''.join(filter(str.isdigit, basename))
    return int(number) if number else 0

def get_class_matrix(matrix):
    mask = np.zeros(matrix.shape, np.float64)
    conditions = [
        matrix == 69,
        matrix == 109,
        matrix == 153,
        matrix == 204,
        matrix == 255, 
        matrix == 127,
        matrix == 50,
    ]
    values = [1, 2, 3, 4, 5, -1, -1]
    for condition, value in zip(conditions, values):
        mask[condition] = value
    return mask

def main():
    args = parse_args()
    # config = load_config(args.config)

    print(f'Calculating the total error for each slice...')

    # experiment_dir, _, metrics_dir = setup_experiment_dirs(config, args.timestamp)

    output_csv = os.path.join(args.save_dir, f"boundary_mistakes.csv")
    mask_dir = args.masks_dir
    pred_path = args.pred_path

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    filenames = sorted(os.listdir(mask_dir), key=extract_number)
    header = ["Image"]
    classes = np.arange(0, 6)

    preds = tifffile.imread(pred_path)

    for i in range(len(classes) - 1):
        header.append(f"Boundary Error ({classes[i]}-{classes[i+1]})")

    for class_id in classes:
        header.append(f"Diff > 1 ({class_id})")

    for class_id in classes:
        header.append(f"Layer {class_id} Pixels")
    
    header.append('Ignore Class Pixels')

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for i, filename in enumerate(filenames):
            mask_path = os.path.join(mask_dir, filename)

            mask_with_ignore = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
   
            mask_with_ignore = cv2.resize(mask_with_ignore, (800, 576)) #TODO: fix
            pred = preds[i]

            mask_with_ignore = get_class_matrix(mask_with_ignore)
            pred = get_class_matrix(pred)
            mask = mask_with_ignore.copy()
            mask[mask_with_ignore == -1] = 0  

            row = [filename]

            for i in range(len(classes) - 1):
                layer1, layer2 = classes[i], classes[i+1]

                G = ((mask == layer1) | (mask == layer2)).astype(np.uint8)
                R_n = ((pred == layer1) & G).astype(np.uint8)
                R_n1 = ((pred == layer2) & G).astype(np.uint8)

                errors_mask_n = ((mask != layer1) & (R_n == 1)).astype(np.uint8)
                errors_mask_n1 = ((mask != layer2) & (R_n1 == 1)).astype(np.uint8)

                errors_on_boundary = errors_mask_n | errors_mask_n1
                errors_on_boundary[mask_with_ignore == -1] = 0  
                error_count = np.sum(errors_on_boundary)
                row.append(error_count)

                if args.debug_dir:
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_G.png"), G * 255)
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_{layer1}-{layer2}_rn.png"), R_n * 255)
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_{layer1}-{layer2}_rn1.png"), R_n1 * 255)
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_{layer1}-{layer2}_errors_mask_n.png"), errors_mask_n * 255)
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_{layer1}-{layer2}_errors_mask_n1.png"), errors_mask_n1 * 255)

            for class_id in classes:
                diff_error_layer = (np.abs(mask - pred) > 1).astype(np.uint8)
                diff_error_layer_class = diff_error_layer * (mask == class_id).astype(np.uint8)
                diff_error_layer_class[mask_with_ignore == -1] = 0  
                diff_error_count_class = np.count_nonzero(diff_error_layer_class)
                row.append(diff_error_count_class)

                if args.debug_dir:
                    cv2.imwrite(os.path.join(args.debug_dir, f"{filename}_{class_id}_diff_error.png"), diff_error_layer_class * 255)

            for class_id in classes:
                layer_pixels = np.count_nonzero(mask == class_id)
                row.append(layer_pixels)

            row.append(np.count_nonzero(mask_with_ignore == -1)) 
            writer.writerow(row)

    print(f'Statistics saved to {output_csv}')

if __name__ == "__main__":
    main()
