import importlib
import argparse
from pathlib import Path
import os
import sys
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import pandas as pd
import numpy as np
import tifffile
import cv2
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))
from file_utils import get_inference_dirs
from residual_unet import ResidualUnet


PIX_FUNCTIONS = {}
OBJ_FUNCTIONS = {}

def get_metric_func(func, isPix=True):
    """
    Retrieve the appropriate metric function based on the metric name and type.

    This function checks whether the specified metric belongs to pixel-based
    or object-based metrics and returns the corresponding function.

    Parameters:
        func (str): The name of the metric to retrieve (e.g., 'iou', 'accuracy').
        isPix (bool): Flag indicating if the metric is pixel-based (True) or object-based (False).

    Returns:
        function: The corresponding metric function if found, otherwise None.
    """

    if isPix:
        if func not in PIX_FUNCTIONS:
            PIX_FUNCTIONS[func] = {
                'iou': smp.metrics.iou_score,
                'accuracy': smp.metrics.accuracy,
                'precision': smp.metrics.precision,
                'recall': smp.metrics.recall,
                'f1': smp.metrics.f1_score,
            }.get(func)
    
    return PIX_FUNCTIONS.get(func) if isPix else OBJ_FUNCTIONS.get(func)

from datetime import datetime

def get_weighted_stats(preds, targets, weights, save_masks=False, save_dir="mask_visualizations"):
    device = preds.device
    num_classes = 6
    
    preds = preds.long() if not isinstance(preds, torch.LongTensor) else preds
    targets = targets.long() if not isinstance(targets, torch.LongTensor) else targets
    
    if weights.dim() == 2:
        weights = weights.unsqueeze(0)  
    weights = weights.to(device).float()
    
    if preds.dim() == 2:
        preds = preds.unsqueeze(0)
        targets = targets.unsqueeze(0)
    
    batch_size = preds.size(0)
    
    tp = torch.zeros((batch_size, num_classes), device=device)
    fp = torch.zeros((batch_size, num_classes), device=device)
    fn = torch.zeros((batch_size, num_classes), device=device)
    tn = torch.zeros((batch_size, num_classes), device=device)
    
    if save_masks:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vis_dir = os.path.join(save_dir, f"masks_{timestamp}")
        os.makedirs(vis_dir, exist_ok=True)
    
    for i in range(batch_size): 
        for cls in range(num_classes):  
            target_cls = (targets[i] == cls)
            pred_cls = (preds[i] == cls)

            tp_mask = (pred_cls & target_cls).float() * weights[i]
            fp_mask = (pred_cls & ~target_cls).float() * weights[i]
            fn_mask = (~pred_cls & target_cls).float() * weights[i]
            tn_mask = (~pred_cls & ~target_cls).float() * weights[i]

            
            # if save_masks:
            #     # Convert masks to uint8 images (0-255)
            #     tp_img = (tp_mask.cpu().numpy() * 255).astype(np.uint8)
            #     fp_img = (fp_mask.cpu().numpy() * 255).astype(np.uint8)
            #     fn_img = (fn_mask.cpu().numpy() * 255).astype(np.uint8)
            #     tn_img = (tn_mask.cpu().numpy() * 255).astype(np.uint8)
                
      
            #     cv2.imwrite(os.path.join(vis_dir, f"img{i}_cls{cls}_tp.png"), tp_img)
            #     cv2.imwrite(os.path.join(vis_dir, f"img{i}_cls{cls}_fp.png"), fp_img)
            #     cv2.imwrite(os.path.join(vis_dir, f"img{i}_cls{cls}_fn.png"), fn_img)
            #     cv2.imwrite(os.path.join(vis_dir, f"img{i}_cls{cls}_tn.png"), tn_img)
                
       
            #     if cls == 0:  # Save only once per image
            #         pred_img = (preds[i].cpu().numpy() * (255/num_classes)).astype(np.uint8)
            #         target_img = (targets[i].cpu().numpy() * (255/num_classes)).astype(np.uint8)
            #         cv2.imwrite(os.path.join(vis_dir, f"img{i}_pred.png"), pred_img)
            #         cv2.imwrite(os.path.join(vis_dir, f"img{i}_target.png"), target_img)

                  
            tp[i, cls] = torch.sum(tp_mask)
            fp[i, cls] = torch.sum(fp_mask)
            fn[i, cls] = torch.sum(fn_mask)
            # tn[i, cls] = torch.sum(tn_mask)
    
    return tp, fp, fn, tn

def compute_weighted_metrics(preds, targets, weights, config):
    device = preds.device
    metrics = {}
    
    try:        
        if preds.dim() == 3:
            preds = preds.unsqueeze(0)
            targets = targets.unsqueeze(0)
            
        
        if weights is not None:
            weights = weights.unsqueeze(0)
            weights = weights.to(device).float() if torch.is_tensor(weights) else torch.tensor(weights, device=device).float()
            tp, fp, fn, tn = get_weighted_stats(preds, targets, weights)
        else:
            tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, ignore_index=config.SegmentationClass.IGNORE_INDEX.value, num_classes=config.NUM_CLASSES, mode=config.LOSS_MODE)
        
        for metric_name in config.SEGM_METRICS:
            func = get_metric_func(metric_name)
            if func is None:
                continue
                
            for reduction in config.REDUCTION_TYPES:
                if reduction == 'object':
                    continue
                    
                try:
                    if reduction == 'micro-imagewise':
                        flat_tp = tp.sum().unsqueeze(0)
                        flat_fp = fp.sum().unsqueeze(0)
                        flat_fn = fn.sum().unsqueeze(0)
                        metric_val = func(flat_tp, flat_fp, flat_fn, None, 'micro')
                    elif reduction in ['weighted', 'weighted-imagewise']:
                        metric_val = func(tp, fp, fn, tn, reduction.split('-')[0], 
                                       class_weights=config.CLASS_WEIGHTS)
                    else:
                        metric_val = func(tp, fp, fn, tn, reduction)
                    
                    if metric_val is not None:
                        key = f"{metric_name}_{reduction.replace('-', '_')}"
                        metrics[key] = metric_val.item() if torch.is_tensor(metric_val) else metric_val
                except Exception as e:
                    print(f"Error computing {metric_name} with {reduction}: {e}")
                    metrics[f"{metric_name}_{reduction.replace('-', '_')}"] = float('nan')

                    
        return metrics
        
    except Exception as e:
        print(f"Error in compute_weighted_metrics: {e}")
        raise

def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation errors.")
    parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
    parser.add_argument("--timestamp", required=True, help="Experiment timestamp")
    parser.add_argument("--epoch", default=None, help="Experiment epoch")
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

def load_model(checkpoint_path, device, config):
    model = ResidualUnet(config.INPUT_CHANNELS, config.NUM_CLASSES, config.CH_AGGR_TYPE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval()
    return model

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

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch = args.epoch

    experiment_dir, log_dir, metrics_dir = setup_experiment_dirs(config, args.timestamp)
    csv_path = metrics_dir / f"{args.timestamp}_weighted_metrics_{epoch}.csv"
    img_csv_path = metrics_dir / f"{args.timestamp}_weighted_metrics_img_{epoch}.csv"

    if config.LOSS_WEIGHTS_PATH:
        consistency_map = tifffile.imread(config.LOSS_WEIGHTS_PATH)
        weights = (consistency_map > 0.5).astype(np.uint8)
        # weights = consistency_map
       
    else:
        weights = None

    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((config.INPUT_HEIGHT, config.INPUT_WIDTH), 
                         interpolation=transforms.InterpolationMode.NEAREST)
    ])  

    results = []
    img_results = []
    modes = ['test', 'val', 'train']

    weights_indices ={'test': [0, 40], 'val': [40, 60], 'train': [60, 300]}

    os.makedirs("pred", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("mistakes", exist_ok=True)

    print(f'Calculating metrics for each epoch...')

    for mode in modes:
        imgs_dir, masks_dir, _ = get_inference_dirs(mode, config)
        imgs_filenames = sorted(os.listdir(imgs_dir), key=extract_number)
        masks_filenames = sorted(os.listdir(masks_dir), key=extract_number)
        
        checkpoint_files = sorted([f for f in os.listdir(experiment_dir) if f.endswith(".pth")], key=extract_number)

        cur_weights_idx = weights_indices[mode]

        if not epoch:
            selected_checkpoints = checkpoint_files #[f for f in checkpoint_files if extract_number(f) % args.epoch_step == 0]
        else:
            ckpt_basename = checkpoint_files[0].split('_')[0] #  os.path.basename(checkpoint_files[0]).split('_')[0]
            ckpt_filename = f'{ckpt_basename}_{epoch}.pth'
            if ckpt_filename in checkpoint_files:
                selected_checkpoints = [ckpt_filename]
            else:
                raise FileNotFoundError(f"Checkpoint file {ckpt_filename} not found in {checkpoint_files}")

        for checkpoint in selected_checkpoints:
            epoch = checkpoint.split('_')[-1].split('.')[0]
            model = load_model(os.path.join(experiment_dir, checkpoint), device, config)

            print(f"Mode: {mode}, Epoch: {epoch}, checkpoint: {checkpoint}")

            preds, targets = [], []

            for filename in imgs_filenames:
                img_path = os.path.join(imgs_dir, filename)
                mask_path = os.path.join(masks_dir, filename.replace('.tif', '.png'))
                img = tifffile.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = preprocess_mask(mask, config)

                img = transform(img).to(device)
                mask = transform(mask).to(device).long()
                pred = predict_mask(model, img, device)
                
                preds.append(pred)
                targets.append(mask)

            preds = torch.cat(preds, dim=0)
            targets = torch.cat(targets, dim=0)
            weights_tensor = None
            if weights is not None:
                weights_tensor = torch.from_numpy(weights).to(device) 
                weights = weights * (targets != config.SegmentationClass.IGNORE_INDEX) #.float()
               
                metrics = compute_weighted_metrics(preds, targets, weights_tensor[cur_weights_idx[0], cur_weights_idx[1]], config)
            else:
                metrics = compute_weighted_metrics(preds, targets, None, config)
            
            # Prepare row for main CSV
            row = [mode, epoch]
            for metric_name in config.SEGM_METRICS:
                for reduction in config.REDUCTION_TYPES:
                    if reduction != 'object':
                        key = f"{metric_name}_{reduction.replace('-', '_')}"
                        row.append(metrics.get(key, float('nan')))
            
            results.append(row)

            # # Prepare per-image results
            for i, filename in enumerate(imgs_filenames):
                img_row = [epoch, filename]
                
                img_metrics = compute_weighted_metrics(preds[i], targets[i], weights_tensor, config)
                print(img_metrics)
                for metric_name in img_metrics.keys(): #config.SEGM_METRICS:
                    print(metric_name)
                    
                    img_row.append(img_metrics.get(metric_name, float('nan')))
                    print(img_row)
                img_results.append(img_row)

    # Write main metrics CSV
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
   
        header = ["Mode", "Epoch"]
        for metric_name in config.SEGM_METRICS:
            for reduction in config.REDUCTION_TYPES:
                if reduction != 'object':
                    header.append(f"{metric_name}_{reduction}")
        writer.writerow(header)
        writer.writerows(results)

    # Write per-image metrics CSV
    with open(img_csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        img_header = ["Epoch", "Image"]
        for metric_name in config.SEGM_METRICS:
            img_header.append(metric_name)
        writer.writerow(img_header)
        writer.writerows(img_results)
    print("Saved to: ", csv_path)

if __name__ == "__main__":
    main()