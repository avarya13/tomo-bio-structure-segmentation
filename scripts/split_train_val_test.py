import sys
import os
import argparse
import importlib
from shutil import copyfile
import torch
from torch.utils.data import random_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

def copy_files(src_dir, dest_dir, files):
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        src_file_path = os.path.join(src_dir, file)
        dest_file_path = os.path.join(dest_dir, file)
        copyfile(src_file_path, dest_file_path)

def extract_number(filename):
    number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(number) if number else 0

def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    split_method = args.split_method

    try:
        config = importlib.import_module(args.config)
    except ImportError:
        print(f"Configuration module '{args.config}' not found in 'configs' directory.")
        sys.exit(1)

    train_dir = os.path.join(save_dir, 'train')
    val_dir = os.path.join(save_dir, 'val')
    test_dir = os.path.join(save_dir, 'test')

    folders = ['x', 'y']

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        data = sorted(os.listdir(folder_path), key=extract_number)
  

        if split_method == 'seq':
            train_data = data[config.TRAIN_SLICES['start']:config.TRAIN_SLICES['end']]
            val_data = data[config.VAL_SLICES['start']:config.VAL_SLICES['end']]
            test_data = data[config.TEST_SLICES['start']:config.TEST_SLICES['end']]

        elif split_method == 'uni':
            train_data = data[config.VAL_SLICES['start']:config.TRAIN_SLICES['end']]

            val_data = [train_data[i] for i in range(len(train_data)) if i % config.VAL_INTERVAL == 0]

            test_data = data[config.TEST_SLICES['start']:config.TEST_SLICES['end']]

            train_data = [file for file in train_data if file not in val_data and file not in test_data]

        elif split_method == 'rnd':
            train_data, val_data, test_data = random_split(dataset=data, lengths=[config.TRAIN_SLICES_LEN, config.VAL_SLICES_LEN, config.TEST_SLICES_LEN], generator=torch.Generator().manual_seed(config.SEED))

        else:
            print(f"Unknown split method '{split_method}'. Please use 'rnd', 'seq' or 'uni'.")
            sys.exit(1)

        copy_files(folder_path, os.path.join(train_dir, folder), train_data)
        copy_files(folder_path, os.path.join(val_dir, folder), val_data)
        copy_files(folder_path, os.path.join(test_dir, folder), test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize dataset into train, validation, and test directories.")
    
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the source data directory.')
    parser.add_argument('--save_dir', type=str, required=True, 
                        
                        help='Path to the directory where the split data will be saved.')
    parser.add_argument('--config', type=str, default='layers_seq_ave_mid_inv', 
                        help='Configuration module name.')
    parser.add_argument('--split_method', type=str, choices=['seq', 'uni', 'rnd'], default='seq',
                        help="Method for splitting data.")

    args = parser.parse_args()
    main(args)
