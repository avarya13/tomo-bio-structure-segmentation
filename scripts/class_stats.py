import argparse
import importlib
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))


def compute_pixels_stats(img_dir):
    class_0_total, class_1_total = 0, 0
    total_pixels = 0

    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        img = np.array(Image.open(img_path))

        # Count pixels for classes
        class_0_total += (img == 0).sum()
        class_1_total += (img == 255).sum()

        # Update total pixel count
        total_pixels += img.size

        """ # Optional: Debugging output
        print(f"File: {file}")
        print(f"Image size: {img.size} pixels")
        print(f"Current total_pixels: {total_pixels}")
        print(f"Current class_0_total: {class_0_total}")
        print(f"Current class_1_total: {class_1_total}") """

    if total_pixels == 0:
        raise ValueError("No pixels found in the provided directory or images are empty.")
    
    # Calculate proportions as decimals
    class_0_proportion = class_0_total / total_pixels
    class_1_proportion = class_1_total / total_pixels

    return class_0_total, class_1_total, class_0_proportion, class_1_proportion



def main(args):
    config = importlib.import_module(args.config)
    timestamp = args.timestamp
    epoch = args.epoch
    mode = args.mode

    metrics_dir = os.path.join(config.EXPERIMENTS_DIR, timestamp, 'metrics')

    if not os.path.exists(metrics_dir):
        print(f'Path {metrics_dir} does not exist. Run inference for the experiment {timestamp} (epoch {epoch}) first.')
        return

    data_types = ['train', 'val', 'test'] if mode == 'all' else [mode]
    

    all_data = []

    for data_type in data_types:
        visual_dir = os.path.join(config.EXPERIMENTS_DIR, f'{timestamp}', f'{data_type}_visualization', f'{timestamp}_{epoch}')
        pred_dir = os.path.join(visual_dir, 'predictions')
        masks_dir = os.path.join(visual_dir, 'masks')

        skip_data_type = False

        for img_dir in [pred_dir, masks_dir]:
            if not os.path.exists(img_dir):
                print(f'Path {img_dir} does not exist. Run inference for the experiment {timestamp} (epoch {epoch}) on {data_type} set first.')
                skip_data_type = True
                break  

        if skip_data_type:
            continue  

        # Compute pixel statistics for masks and predictions
        masks_class_0_total, masks_class_1_total, masks_class_0_proportion, masks_class_1_proportion = compute_pixels_stats(masks_dir)
        pred_class_0_total, pred_class_1_total, pred_class_0_proportion, pred_class_1_proportion = compute_pixels_stats(pred_dir)
        
        # Append data for each statistic and class
        all_data.extend([
            {'dataset': data_type, 'stat': 'gt_count', 'class_0': masks_class_0_total, 'class_1': masks_class_1_total},
            {'dataset': data_type, 'stat': 'pred_count', 'class_0': pred_class_0_total, 'class_1': pred_class_1_total},
            {'dataset': data_type, 'stat': 'gt_proportion', 'class_0': masks_class_0_proportion, 'class_1': masks_class_1_proportion},
            {'dataset': data_type, 'stat': 'pred_proportion', 'class_0': pred_class_0_proportion, 'class_1': pred_class_1_proportion}
        ])

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    csv_file_path = os.path.join(metrics_dir, f'pix_stats_{timestamp}_{epoch}.csv')
    df.to_csv(csv_file_path, index=False)

    print(f'Pixel statistics for each class saved to {csv_file_path}.')

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Olfactory bulb segmentation project')
    parser.add_argument('--timestamp', type=str, required=True, help='The label of the experiment in the format: YYYYMMDD-HHMM')
    parser.add_argument('--epoch', type=int, required=True, help='Specific epoch to test. Optional for single_inference.') 
    parser.add_argument('--config', type=str, required=True, help="Name of the config file to use without the .py extension")
    parser.add_argument('--mode', type=str, choices=['test', 'val', 'train', 'all'], default='all', help='Mode of operation: test, validation, or all (for full_inference)')

    args = parser.parse_args()
    main(args)
