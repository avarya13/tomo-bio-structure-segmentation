import os
import sys
import argparse
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd

def calc_obj_diff(prev_mask, new_mask):
    num_labels_prev, _ = cv2.connectedComponents(prev_mask)
    num_labels_new, _ = cv2.connectedComponents(new_mask)

    """ obj_diff = 0

    for label_target in range(1, num_labels_prev): 
            mask1 = (labels_prev == label_target)
            
            for label_new in range(1, num_labels_new): 
                mask2 = (labels_new == label_new)

                if torch.any(mask1 & mask2):
                    if not matched_pred[label_pred]: """

    return abs(num_labels_new - num_labels_prev)


def main(args):
    prev_files = os.listdir(args.prev_dir)
    new_files = os.listdir(args.new_dir)

    pix_diff, obj_diff = 0, 0

    diff_df = pd.DataFrame(columns=['filename', 'pix_diff', 'obj_diff'])

    for prev_file, new_file in zip(prev_files, new_files):
        prev_mask =  np.array(Image.open(os.path.join(args.prev_dir, prev_file)))
        new_mask = np.array(Image.open(os.path.join(args.new_dir, new_file)))

        filename = os.path.splitext(prev_file)[0]

        cur_pix_diff = (prev_mask != new_mask).sum()

        cur_obj_diff = calc_obj_diff(prev_mask, new_mask)

        diff_df.loc[len(diff_df.index)] = [filename, cur_pix_diff, cur_obj_diff]

        pix_diff += cur_pix_diff

        obj_diff += cur_obj_diff

    diff_df.loc[len(diff_df.index)] = ['total', pix_diff, obj_diff]

    diff_df.to_csv('masks_diff.csv', index=False)        

    print(pix_diff, obj_diff)



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Difference between old and new masks.')
    parser.add_argument('--prev_dir', default=r'D:\datasets\луковица\experiments_data\60_to_300_masks\thresholded', type=str, help='The label of the experiment in the format: YYYYMMDD-HHMM')
    parser.add_argument('--new_dir', default=r'D:\datasets\луковица\experiments_data\v2_fixed\interpolated\thresholded', type=str, help='Specific epoch to test. Optional for single_inference.') 
    
    args = parser.parse_args()
    main(args)
