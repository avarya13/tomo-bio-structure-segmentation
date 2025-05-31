import os
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2

pred_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/experiments_ave_w5s2_bg_30_30_240/20250505-0024/20250505-0024_1000/predictions"
masks_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/masks.binary.roi"

preds_files = sorted(os.listdir(pred_dir))
masks_files = sorted(os.listdir(masks_dir))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preds = []
masks = []

for i in range(len(preds_files)):
    pred_path = os.path.join(pred_dir, preds_files[i])
    mask_path = os.path.join(masks_dir, masks_files[i])

    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    pred = (pred > 0).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    preds.append(torch.from_numpy(pred))
    masks.append(torch.from_numpy(mask))

preds = torch.stack(preds).to(device)
masks = torch.stack(masks).to(device)

tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, num_classes=1, mode='binary')
f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro')

print("F1-score:", f1.item())
