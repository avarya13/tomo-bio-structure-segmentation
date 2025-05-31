import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.bce_config import CROPPING_REGIONS, ROI_WIDTH, ROI_HEIGHT, TOTAL_SLICES

def roi_intersection(rect1, rect2):
    (x1_left, y1_top), (x1_right, y1_bottom) = rect1['top_left'], rect1['bottom_right']
    (x2_left, y2_top), (x2_right, y2_bottom) = rect2['top_left'], rect2['bottom_right']

    x_left = max(x1_left, x2_left) 
    y_top = min(y1_top, y2_top)
    x_right = min(x1_right, x2_right)
    y_bottom = max(y1_bottom, y2_bottom)

    if x_left < x_right and y_bottom > y_top:
        return {'top_left': (x_left, y_top), 'bottom_right': (x_right, y_bottom)}
    else:
        return None 

def main():
    bu_dir = r"D:\datasets\луковица\experiments_data\v2\interpolated_v2\300_slices_masks_inter_v2"

    inter_coords = roi_intersection(CROPPING_REGIONS['center'], CROPPING_REGIONS['top-right'])

    if inter_coords is None:
        print("No intersection.")
        return

    width = inter_coords['bottom_right'][0] - inter_coords['top_left'][0]
    height = inter_coords['bottom_right'][1] - inter_coords['top_left'][1]
    inter_area = width * height

    bu_total = 0
    bg_total = 0

    no_total = TOTAL_SLICES * (ROI_WIDTH * ROI_HEIGHT - inter_area)  
    bg_total += no_total  

    for file in os.listdir(bu_dir):
        img = np.array(Image.open(os.path.join(bu_dir, file)))  
        
        bu_total += (img == 255).sum()
        
        bg_total += (img == 0).sum()

    total_pixels = bu_total + bg_total 

    print(inter_area, total_pixels)

    bu_freq = bu_total / total_pixels if total_pixels > 0 else 0
    bg_freq = bg_total / total_pixels if total_pixels > 0 else 0

    print(bu_total, bg_total)
    print(f"bu share: {bu_freq:.4f}, bg share: {bg_freq:.4f}")
    print(f"Sum of shares: {bu_freq + bg_freq:.4f}")

    bu_weight = 1 / bu_freq if bu_freq > 0 else 0
    bg_weight = 1 / bg_freq if bg_freq > 0 else 0

    print(bu_weight, bg_weight)

    norm_bu_weight = bu_weight / (bu_weight + bg_weight)
    norm_bg_weight = bg_weight / (bu_weight + bg_weight)

    print(f"Normalized weights: bu={norm_bu_weight:.4f}, bg={norm_bg_weight:.4f}")

if __name__ == "__main__":
    main()