import importlib
import os
import torch
import segmentation_models_pytorch as smp
import pandas as pd
import numpy as np
import cv2

PIX_FUNCTIONS = {}
OBJ_FUNCTIONS = {}

def get_metric_func(func, is_pix=True):
    """
    Retrieve the appropriate metric function based on the metric name and type.

    This function checks whether the specified metric belongs to pixel-based
    or object-based metrics and returns the corresponding function.

    Parameters:
        func (str): The name of the metric to retrieve (e.g., 'iou', 'accuracy').
        is_pix (bool): Flag indicating if the metric is pixel-based (True) or object-based (False).

    Returns:
        function: The corresponding metric function if found, otherwise None.
    """

    if is_pix:
        if func not in PIX_FUNCTIONS:
            PIX_FUNCTIONS[func] = {
                'iou': smp.metrics.iou_score,
                'accuracy': smp.metrics.accuracy,
                'precision': smp.metrics.precision,
                'recall': smp.metrics.recall,
                'f1': smp.metrics.f1_score,
            }.get(func)
    else:
        if func not in OBJ_FUNCTIONS:
            OBJ_FUNCTIONS[func] = {
                'precision': obj_precision,
                'recall': obj_recall,
            }.get(func)
    
    return PIX_FUNCTIONS.get(func) if is_pix else OBJ_FUNCTIONS.get(func)


def obj_precision(tp, fp, **kwargs):
    """
    Calculate object-level precision.

    Precision is defined as the ratio of true positives to the sum of true positives and false positives.

    Parameters:
        tp (int): Number of true positives.
        fp (int): Number of false positives.

    Returns:
        float: Precision value (0 if no positive predictions).
    """
    return tp / (tp + fp) if tp + fp > 0 else 0

def obj_recall(tp, fn, **kwargs):
    """
    Calculate object-level recall.

    Recall is defined as the ratio of true positives to the sum of true positives and false negatives.

    Parameters:
        tp (int): Number of true positives.
        fn (int): Number of false negatives.

    Returns:
        float: Recall value (0 if no actual positives).
    """
    return tp / (tp + fn) if tp + fn > 0 else 0


def get_object_stats(targets, predictions, device, logger):
    """
    Compute true positives, false positives, and false negatives for object-based metrics.

    This function calculates statistics for the objects detected in the predicted segmentation masks
    compared to the ground truth masks.

    Parameters:
        targets (torch.Tensor): Ground truth segmentation masks.
        predictions (torch.Tensor): Predicted segmentation masks.
        device (torch.device): Device to which tensors should be moved for computation.
        logger (logging.Logger): Logger instance for recording debug information.

    Returns:
        tuple: (true positives, false positives, false negatives).
    """

    num_images = targets.size(0) if targets.dim() > 2 else 1
    total_objects = 0 
    tp, fp, fn = 0, 0, 0 

    if targets.dim() == 4:
        targets = targets.squeeze(1)
        predictions = predictions.squeeze(1)

    for i in range(num_images):
        if targets.dim() > 2:
            target = targets[i].cpu().numpy().astype(np.uint8)
            prediction = predictions[i].cpu().numpy().astype(np.uint8)
        else:
            target = targets.cpu().numpy().astype(np.uint8)
            prediction = predictions.cpu().numpy().astype(np.uint8)        

        num_labels_target, labels_target = cv2.connectedComponents(target)
        num_labels_pred, labels_pred = cv2.connectedComponents(prediction)

        total_objects += num_labels_target - 1

        labels_target = torch.tensor(labels_target, device=device, dtype=torch.int32)
        labels_pred = torch.tensor(labels_pred, device=device, dtype=torch.int32)

        match_targets = torch.zeros(num_labels_target, dtype=torch.bool, device=device)
        matched_pred = torch.zeros(num_labels_pred, dtype=torch.bool, device=device)

        for label_target in range(1, num_labels_target): 
            mask1 = (labels_target == label_target)
            
            for label_pred in range(1, num_labels_pred): 
                mask2 = (labels_pred == label_pred)

                if torch.any(mask1 & mask2):
                    if not matched_pred[label_pred]:
                        tp += 1
                        match_targets[label_target] = True
                        matched_pred[label_pred] = True 
                    break
                
        fp += torch.sum(~matched_pred[1:]).item()
        fn += torch.sum(~match_targets[1:]).item()
        
    """ logger.info(f'Total Objects (excluding background): {total_objects}')
    logger.info(f'True Positives (TP): {tp}')
    logger.info(f'False Positives (FP): {fp}')
    logger.info(f'False Negatives (FN): {fn}') """

    return tp, fp, fn


def compute_metrics(preds, targets, config, logger):
    """
    Compute various segmentation metrics based on model predictions and ground truth targets.

    This function calculates pixel-based and object-based metrics by comparing the predictions against
    the target segmentation masks and returns the results in a dictionary.

    Parameters:
        preds (torch.Tensor): Predicted segmentation masks.
        targets (torch.Tensor): Ground truth segmentation masks.
        config (object): Configuration object containing settings for metric computation.
        logger (logging.Logger): Logger instance for recording metric computation details.

    Returns:
        dict: A dictionary containing computed metrics.
    
    Raises:
        Exception: Logs an error message and raises an exception if metric computation fails.
    """

    device = preds.device

    if not isinstance(preds, torch.LongTensor):
        preds = preds.long()
    if not isinstance(targets, torch.LongTensor):
        targets = targets.long()

    # preds[preds == 0] = config.SegmentationClass.IGNORE_INDEX.value
    # targets[targets == 0] = config.SegmentationClass.IGNORE_INDEX.value

    metrics = {}  

    # print('ignore', config.SegmentationClass.IGNORE_INDEX.value)

    try:
       
        if config.LOSS_MODE == 'binary':
            tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, num_classes=config.NUM_CLASSES, mode=config.LOSS_MODE)
        else:
            tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, ignore_index=config.SegmentationClass.IGNORE_INDEX.value, num_classes=config.NUM_CLASSES, mode=config.LOSS_MODE)

        
        for metric_name in config.SEGM_METRICS:
            func = get_metric_func(metric_name)
            if func:
                for reduction in config.REDUCTION_TYPES:
                    if reduction != 'object':
                        if reduction=='weighted' or reduction=='weighted-imagewise':
                            metric_val = func(tp.to(device), fp.to(device), fn.to(device), tn.to(device), reduction, class_weights=config.CLASS_WEIGHTS)
                        else:
                            metric_val = func(tp.to(device), fp.to(device), fn.to(device), tn.to(device), reduction)
                        metrics[f"{metric_name}_{reduction.replace('-', '_')}"] = metric_val.item()

        
        if 'object' in config.REDUCTION_TYPES:
            object_tp, object_fp, object_fn = get_object_stats(targets, preds, device, logger)
            for metric_name in config.OBJECT_METRICS:
                obj_func = get_metric_func(metric_name, is_pix=False)
                if obj_func:
                    metric_val = obj_func(tp=object_tp, fp=object_fp, fn=object_fn)
                    metrics[f"{metric_name}_object"] = metric_val

        #save_metrics_to_csv(metrics, "metrics.csv", logger)

        return metrics

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise


def log_test_metrics(metrics, logger):
    """
    Log the computed metrics for testing.

    This function formats and logs each metric from the provided metrics dictionary.

    Parameters:
        metrics (dict): Dictionary containing metric names and their values.
        logger (logging.Logger): Logger instance for recording metrics.
    """

    for metric in metrics:
        metric_name = metric.replace('_', ' ', 1).replace('_', '-', 2).title() 
        logger.info(f"{metric_name}: {metrics[metric]:.4f}")


def summarize_metrics(metrics, model_name, metrics_file, logger):
    """
    Summarize the computed metrics and save them to a CSV file.

    This function creates a row of metrics data, logs the metrics, and appends them to a CSV file.

    Parameters:
        metrics (dict): Dictionary containing the computed metrics.
        model_name (str): Name of the model to associate with the metrics.
        metrics_file (str): Path to the CSV file where metrics should be saved.
        logger (logging.Logger): Logger instance for recording operations.
    
    Raises:
        Exception: Logs an error message and raises an exception if saving metrics fails.
    """

    try:
        metrics_row = {'model': model_name}
        metrics_row.update(metrics)

        log_test_metrics(metrics, logger)

        metrics_df = pd.DataFrame.from_dict([metrics_row])

        def extract_epoch(model_name):
            return int(model_name.split('-')[-1])  

        if os.path.exists(metrics_file):
            existing_df = pd.read_csv(metrics_file)
            
            combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)

            combined_df['epoch'] = combined_df['model'].apply(extract_epoch)
            combined_df = combined_df.sort_values(by='epoch', ascending=True)  
            combined_df.drop(columns=['epoch'], inplace=True)  
            
            combined_df.to_csv(metrics_file, index=False)
        else:
            metrics_df.to_csv(metrics_file, index=False)

        logger.info(f"Metrics added to {metrics_file}")

    except Exception as e:
        logger.error(f"Failed to add metrics to {metrics_file}. Exception: {e}")
        raise


def compute_imagewise_metrics(preds, targets, filenames, save_path, config, logger):
    """
    Compute metrics for each image and save them to a CSV file.

    This function iterates over each predicted and target mask, computes metrics,
    and saves the results in a structured format.

    Parameters:
        preds (torch.Tensor): Predicted segmentation masks.
        targets (torch.Tensor): Ground truth segmentation masks.
        filenames (list): List of image filenames corresponding to the predictions and targets.
        save_path (str): Path where the metrics CSV file should be saved.
        config (object): Configuration object for metric computation settings.
        logger (logging.Logger): Logger instance for recording operations.

    Raises:
        Exception: Logs an error message and raises an exception if metric computation or saving fails.
    """

    try:
        metrics_list = []
        for pred, target, filename in zip(preds, targets, filenames):
            metrics = compute_metrics(pred.unsqueeze(0), target.unsqueeze(0), config, logger)
            metrics_flat = {'image': filename}  
            metrics_flat.update(metrics)            
            metrics_list.append(metrics_flat)
        
        if metrics_list:
            df = pd.DataFrame(metrics_list)
            cols = ["image"] + [col for col in df.columns if col != "image"]
            df = df[cols]
            #df.sort_values(by='iou_micro', ascending=True, inplace=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Metrics for each image saved to {save_path}.")
        else:
            logger.info("No metrics to save for each image.")
    except Exception as e:
        logger.error(f"Failed to create metrics table. Exception: {e}")
        raise


def calculate_metrics_by_quarters(metrics_path, model_name, mode):
    """
    Calculate and save metrics averaged over quarters from a CSV file.

    This function divides the metrics data into four equal quarters and calculates
    the average for each metric within those quarters. The results are saved to a new CSV file.

    Parameters:
        metrics_path (str): Path to the directory containing metrics CSV files.
        model_name (str): Name of the model to associate with the metrics.
        mode (str): Mode indicating the type of metrics (e.g., 'train', 'test').
    """

    img_df = pd.read_csv(os.path.join(metrics_path, f'img_{mode}_{model_name}.csv'))
    
    bu_df = img_df[img_df['image'].str.startswith('bu')]
    no_df = img_df[img_df['image'].str.startswith('no')]
    
    def calculate_quarters_for_group(df, group_name):
        len_quarter = len(df) // 4
        quarters = [
            df.iloc[:len_quarter],
            df.iloc[len_quarter:2 * len_quarter],
            df.iloc[2 * len_quarter:3 * len_quarter],
            df.iloc[3 * len_quarter:]
        ]
        
        quarter_df = pd.DataFrame(columns=['metric', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4'])
        
        for col in df.columns[1:]:  
            row = {'metric': col}
            for i, quarter in enumerate(quarters):
                row[f'quarter_{i+1}'] = quarter[col].mean()  
            quarter_df.loc[len(quarter_df.index)] = row
        
        quarter_df.to_csv(os.path.join(metrics_path, f'quart_{group_name}_{mode}_{model_name}.csv'), index=False)

    calculate_quarters_for_group(bu_df, 'bu')
    calculate_quarters_for_group(no_df, 'no')

""" def calculate_metrics_by_quarters(metrics_path, model_name, mode):
    img_df = pd.read_csv(os.path.join(metrics_path, f'img_{mode}_{model_name}.csv'))
    
    len_quarter = len(img_df) // 4

    quarter_df = pd.DataFrame(columns=['metric', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4'])

    quarters = [
        img_df.iloc[:len_quarter],
        img_df.iloc[len_quarter:2 * len_quarter],
        img_df.iloc[2 * len_quarter:3 * len_quarter],
        img_df.iloc[3 * len_quarter:]
    ]  

    for col in img_df.columns[1:]:  
        row = {'metric': col}  
        for i, quarter in enumerate(quarters):
            row[f'quarter_{i+1}'] = quarter[col].mean()  
        quarter_df.loc[len(quarter_df.index)] = row

    quarter_df.to_csv(os.path.join(metrics_path, f'quart_{mode}_{model_name}.csv'), index=False) """



