import os
import sys
import torch
from torch.utils.data import DataLoader
import importlib
import csv
from pathlib import Path
import segmentation_models_pytorch as smp
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

from predict import predict
from dataset import OlfBulbDataset
from file_utils import get_inference_dirs, get_augmentation
from utils import get_model
from logging_setup import setup_logger

def setup_experiment_dirs(config, timestamp):
    experiment_dir = Path(config.EXPERIMENTS_DIR) / timestamp
    log_dir = experiment_dir / 'logs'
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir, log_dir

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
parser.add_argument('--timestamp', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

config = importlib.import_module(args.config)
timestamp = args.timestamp
model_name = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
experiment_dir, log_dir = setup_experiment_dirs(config, timestamp)
metrics_dir = experiment_dir / 'metrics'
metrics_dir.mkdir(parents=True, exist_ok=True)

logger = setup_logger('test_logger', log_dir / f"test_{timestamp}.log", to_console=True)
model = get_model(model_name, config.INPUT_CHANNELS, config.NUM_CLASSES, config.CH_AGGR_TYPE, device)

mean_metrics_csv = metrics_dir / f"dice_per_classes_{timestamp}.csv"
with open(mean_metrics_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['epoch', 'mode', 'mean_dice'] + [f'dice_cls{i}' for i in range(config.NUM_CLASSES)]
    writer.writerow(header)

models_dir = Path(config.EXPERIMENTS_DIR) / timestamp
checkpoint_files = sorted(models_dir.glob("*.pth"))

for i, checkpoint_path in enumerate(checkpoint_files): #TODO: remove 1000
    if i != 99:
        continue
    logger.info(f"Running inference for checkpoint {checkpoint_path.stem}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    parts = checkpoint_path.stem.split('_')
    epoch = parts[-1] if parts[-1].isdigit() else 'unknown'

    for mode in ['train', 'val', 'test']:
        data_dir, masks_dir, save_aug_dir = get_inference_dirs(mode, config)
        augmentation = get_augmentation(mode, config)

        dataset = OlfBulbDataset(
            data_dir, masks_dir, config,
            save_augmented_dir=save_aug_dir,
            mode=mode,
            augmentation=augmentation,
            inference=True
        )
        loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=(mode == 'train'),
            num_workers=4
        )

        masks, preds = predict(model, loader, device, config, logger)
        preds = preds.long()
        targets = masks.long()

        class_metrics = {cls: [] for cls in range(config.NUM_CLASSES)}
        ignore_index = config.SegmentationClass.IGNORE_INDEX.value

        for cls in range(config.NUM_CLASSES):
            preds_cls = (preds == cls).long()
            targets_cls = (targets == cls).long()

            tp, fp, fn, tn = smp.metrics.get_stats(
                preds_cls,
                targets_cls,
                ignore_index=ignore_index,
                num_classes=2,
                mode=config.LOSS_MODE
            )

            dice = smp.metrics.f1_score(
                tp.to(device),
                fp.to(device),
                fn.to(device),
                tn.to(device),
                reduction='macro'
            )

            class_metrics[cls].append(dice.item())
            logger.info(f'Mode {mode} Class {cls} Dice: {dice:.4f}')

        dice_scores = [v[0] for v in class_metrics.values()]
        mean_dice = torch.tensor(dice_scores).mean().item()
        logger.info(f'Mode {mode} Mean Dice across classes: {mean_dice:.4f}')

        with open(mean_metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, mode, f"{mean_dice:.6f}"] + [f"{v[0]:.6f}" for v in class_metrics.values()])
