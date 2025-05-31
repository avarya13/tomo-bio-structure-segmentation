import os
import sys
import argparse
import importlib
import logging
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

# Append paths
sys.path.append(str(Path(__file__).resolve().parent / 'src'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'configs')))

# Custom imports
import utils
from dataset import OlfBulbDataset
from predict import predict
from metrics import compute_metrics, compute_imagewise_metrics, summarize_metrics, calculate_metrics_by_quarters
from file_utils import get_image_paths, get_model_path, get_inference_dirs, get_augmentation
from utils import get_model
from visualize_results import visualize
from logging_setup import setup_logger

def run_single_inference(mode, model_name, config, timestamp, epoch, visualize_results=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = config.EXPERIMENTS_DIR / timestamp
    log_dir = experiment_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    logger_name = f'test_logger_{timestamp}'
    if not logging.getLogger(logger_name).hasHandlers():
        test_logger = setup_logger(logger_name, log_dir / f"test_{timestamp}.log", to_console=True)
    else:
        test_logger = logging.getLogger(logger_name)

    model_path, model_epoch = get_model_path(experiment_dir, test_logger, epoch=epoch)
    if not model_path:
        return

    model_file = os.path.splitext(os.path.basename(model_path))[0]
    visualization_dir = os.path.join(experiment_dir, f'{timestamp}_{model_epoch}')
    metrics_dir = os.path.join(experiment_dir, 'metrics')
    os.makedirs(visualization_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    test_logger.info(f"Starting inference for the {mode} set...")
    test_logger.info(f"Loading model from: {model_path}...")

    model = get_model(model_name, config.INPUT_CHANNELS, config.NUM_CLASSES, config.CH_AGGR_TYPE, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_logger.info("Model loaded successfully.")

    data_dir, masks_dir, save_augmented_dir = get_inference_dirs(mode, config)
    augmentation = get_augmentation(mode, config)
    shuffle = True if mode == 'train' else False
    dataset = OlfBulbDataset(data_dir, masks_dir, config, save_augmented_dir=save_augmented_dir, mode=mode, augmentation=augmentation, inference=True)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
    filenames = [filename for _, _, filename, _ in dataset]

    masks, predicted = predict(model, loader, device, config, test_logger, visualization_dir)
    test_logger.info(f"{mode.capitalize()} predictions saved to {visualization_dir}.")

    per_img_file = os.path.join(metrics_dir, f'img_{mode}_{model_file}.csv') 
    metrics_file = os.path.join(metrics_dir, f'{mode}_{model_file}.csv') 

    metrics = compute_metrics(predicted, masks, config, test_logger)
    summarize_metrics(metrics, model_file, metrics_file, test_logger)
    compute_imagewise_metrics(predicted, masks, filenames, per_img_file, config, test_logger)         
    calculate_metrics_by_quarters(metrics_dir, model_file, mode)

    if visualize_results:
        original_paths = get_image_paths(visualization_dir, 'slices', test_logger)
        masks_paths = get_image_paths(visualization_dir, 'masks', test_logger)
        predicted_paths = get_image_paths(visualization_dir, 'predictions', test_logger)
        visualize(original_paths, masks_paths, predicted_paths, visualization_dir, config, test_logger)
        test_logger.info(f'{mode.capitalize()} overlays saved to {visualization_dir}.')

def main():
    parser = argparse.ArgumentParser(description='Single inference for olfactory bulb segmentation')
    parser.add_argument('--model', type=str, choices=['unet', 'unet_ref', 'res_unet', 'res_unet_small', 'unet_small'], default='res_unet', help='Model name')
    parser.add_argument('--config', type=str, required=True, help='Config file name (without .py extension)')
    parser.add_argument('--timestamp', type=str, help='Experiment timestamp in format YYYYMMDD-HHMM')
    parser.add_argument('--epoch', type=str, required=True, help='Epoch to load')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], default='test', help='Dataset type')
    parser.add_argument('--visualize_results', action='store_true', help='Visualize predictions')

    args = parser.parse_args()

    config = importlib.import_module(args.config)

    if not args.timestamp:
        args.timestamp = datetime.now().strftime('%Y%m%d-%H%M')

    run_single_inference(args.mode, args.model, config, args.timestamp, int(args.epoch), args.visualize_results)

if __name__ == '__main__':
    main()
