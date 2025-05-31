import importlib
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))


def plot(data, epochs_range, smooth_factor, color, label, ls='-'):
    if data is not None:
        data = data.set_index('Epoch')  
        data.index = pd.to_numeric(data.index, errors='coerce')  
        data = data.reindex(epochs_range).interpolate(method='linear').ewm(alpha=smooth_factor).mean()
        plt.plot(epochs_range, data['Mean Unstable Pixels'], label=label, color=color, linewidth=3, ls=ls)

def plot_train_val(ori_train, ori_val, ave_train, ave_val, config, total_epochs=2000, save_dir=None):
    os.makedirs(save_dir, exist_ok=True)

    epochs_range = ori_train['Epoch'].unique()  
    # print(epochs_range)

    plt.figure(figsize=config.FIGURE_SIZE)  
    smooth_factor = 0.2  # config.SMOOTH_FACTOR

    plot(ori_train, epochs_range, smooth_factor, color='mediumblue', label="ori-train")
    plot(ori_val, epochs_range, smooth_factor, color='green', label="ori-val")
    plot(ave_train, epochs_range, smooth_factor, color='darkmagenta', label="ave-train")
    plot(ave_val, epochs_range, smooth_factor, color='lightseagreen', label="ave-val")
    
    plt.title(f"Smoothness over {total_epochs} epochs (train and val)", fontsize=config.TITLE_FONTSIZE)
    plt.xlabel("Epoch", fontsize=config.LABELS_FONTSIZE, labelpad=config.LABELPAD)
    plt.ylabel("Unstable pixels", fontsize=config.LABELS_FONTSIZE, labelpad=config.LABELPAD)
    plt.legend(loc=config.LEGEND_LOC, fontsize=config.LEGEND_FONTSIZE)
    plt.grid(True)
    
    plt.xticks(np.arange(0, 2000, 200), rotation=0, fontsize=24)  
    plt.yticks(fontsize=24)

    plt.ylim(4000, 40000)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, f"smooth_train_val.{config.SAVE_EXT}")) 
    plt.close()

    print(f"Plotting for the train and val completed. Result was saved to {save_dir}")

def plot_test(ori_test, ave_test, config, total_epochs=2000, save_dir=None):
    os.makedirs(save_dir, exist_ok=True)

    epochs_range = ori_test['Epoch'].unique()  

    plt.figure(figsize=config.FIGURE_SIZE)  
    smooth_factor = 0.2  

    plot(ori_test, epochs_range, smooth_factor, color='crimson', label="ori-test")
    plot(ave_test, epochs_range, smooth_factor, color='darkorange', label="ave-test")
    
    plt.title(f"Smoothness (for model trained without weights)", fontsize=config.TITLE_FONTSIZE)
    plt.xlabel("Epoch", fontsize=config.LABELS_FONTSIZE, labelpad=config.LABELPAD)
    plt.ylabel("Unstable pixels", fontsize=config.LABELS_FONTSIZE, labelpad=config.LABELPAD)
    plt.legend(loc=config.LEGEND_LOC, fontsize=config.LEGEND_FONTSIZE)
    plt.grid(True)
    
    plt.xticks(np.arange(0, 4000, 200), rotation=0, fontsize=24)  
    plt.yticks(fontsize=24)

    # plt.ylim(4000, 14000)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, f"smooth_test.{config.SAVE_EXT}")) 
    plt.close()

    print(f"Plotting for the test completed. Result was saved to {save_dir}")


def plot_train_val_test(ori_train, ori_val, ave_train, ave_val, ori_test, ave_test, config, total_epochs=3000, save_dir='dashed')  :
    os.makedirs(save_dir, exist_ok=True)

    epochs_range = ave_train['Epoch'].unique()  
    print(epochs_range)

    plt.figure(figsize=config.FIGURE_SIZE)  
    smooth_factor = 0.2  # config.SMOOTH_FACTOR

    if ori_train is not None:    
        plot(ori_train, epochs_range, smooth_factor, color='mediumblue', label="train-ave", ls='--')
    plot(ave_train, epochs_range, smooth_factor, color='mediumblue', label="train", ls='-')

    if ori_val is not None: 
        plot(ori_val, epochs_range, smooth_factor, color='green', label="val-ave", ls='--')
    plot(ave_val, epochs_range, smooth_factor, color='green', label="val", ls='-')

    if ori_test is not None: 
        plot(ori_test, epochs_range, smooth_factor, color='crimson', label="test-ave", ls='--')
    plot(ave_test, epochs_range, smooth_factor, color='crimson', label="test", ls='-')

    # plot(ori_train, epochs_range, smooth_factor, color='mediumblue', label="ori-train", ls='--')
    # plot(ori_val, epochs_range, smooth_factor, color='mediumblue', label="ori-val", ls=':')
    # plot(ave_train, epochs_range, smooth_factor, color='green', label="ave-train", ls='--')
    # plot(ave_val, epochs_range, smooth_factor, color='green', label="ave-val", ls=':')

    # plot(ori_test, epochs_range, smooth_factor, color='mediumblue', label="ori-test")
    # plot(ave_test, epochs_range, smooth_factor, color='green', label="ave-test")
    
    plt.title(f"Smoothness (for model trained with confids)", fontsize=34)
    plt.xlabel("Epoch", fontsize=32, labelpad=config.LABELPAD)
    plt.ylabel("Average number of unstable pixels", fontsize=32, labelpad=config.LABELPAD)
    plt.legend(loc='best', fontsize=config.LEGEND_FONTSIZE)
    plt.grid(True)
    
    plt.xticks(np.arange(0, 3000, 500), rotation=0, fontsize=24)  
    plt.yticks(fontsize=24)

    # plt.ylim(5000, 40000)

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()    
    plt.savefig(os.path.join(save_dir, f"smoothness.{config.SAVE_EXT}")) 
    plt.close()

    print(f"Plotting for the train, val, and test completed. Result was saved to {save_dir}")

def parse_csv(path):
    if not path:
        return None, None, None
    data = pd.read_csv(path)
    data['Epoch'] = pd.to_numeric(data['Epoch'], errors='coerce')
    train_data = data[data['Mode'] == 'train'][['Epoch', 'Mean Unstable Pixels']]
    val_data = data[data['Mode'] == 'val'][['Epoch', 'Mean Unstable Pixels']]
    test_data = data[data['Mode'] == 'test'][['Epoch', 'Mean Unstable Pixels']]
    return train_data, val_data, test_data

def main(args):
    config = importlib.import_module(args.config)
    ori_csv = args.ori_csv
    ave_csv = args.ave_csv

    ori_train, ori_val, ori_test = parse_csv(ori_csv)
    ave_train, ave_val, ave_test = parse_csv(ave_csv)

    plot_train_val_test(ori_train, ori_val, ave_train, ave_val, ori_test, ave_test, config, total_epochs=3000, save_dir=args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw epoch-by-epoch performance plots for the retained models in the training, validation, and test sets.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save')
    parser.add_argument('--config', type=str, default='plot_config', help='Path to config module')
    parser.add_argument('--ori_csv', type=str, default=None, help='Path to original CSV file')
    parser.add_argument('--ave_csv', type=str, required=True, help='Path to average CSV file')

    args = parser.parse_args()
    main(args)
