import os
import sys
import argparse
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import tifffile
import csv
from pathlib import Path
from datetime import datetime
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))
from file_utils import get_inference_dirs
from residual_unet import ResidualUnet


def preprocess_mask(mask, config):
    """
    Preprocess the mask by converting colors to class labels.

    Parameters:
        mask (numpy.ndarray): The mask to preprocess.

    Returns:
        numpy.ndarray: The processed mask with class labels.
    """

    mask = mask.astype(np.float64)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation errors per epoch.")
    parser.add_argument("--config", required=True, help="Configuration file name (without .py)")
    return parser.parse_args()


def load_config(config_name):
    config_module = importlib.import_module(config_name)
    return config_module


def extract_number(filename):
    number = ''.join(filter(str.isdigit, os.path.basename(filename)))
    return int(number) if number else 0


def load_mask(path):
    pass


def calculate_statistics(src_mask, enh_mask, config, total_stats):
    """
    Calculates pixel statistics (for both types of annotations) per slice.
    Returns the pixel count for each class, mean pixels per slice, and difference in pixels between masks.
    Also accumulates the total stats across all slices.
    
    Parameters:
        src_mask (numpy.ndarray): The original mask.
        enh_mask (numpy.ndarray): The enhanced (new) mask.
        total_stats (dict): Accumulated statistics across all slices.
    
    Returns:
        dict: Updated total statistics for each class.
    """
    classes = np.unique(enh_mask)
    
    for c in classes:
        src_class_pixels = np.sum(src_mask == c)
        enh_class_pixels = np.sum(enh_mask == c)
        
        if c not in total_stats:
            total_stats[c] = {
                "total_src_pixels": 0,
                "total_enh_pixels": 0,
                "total_diff": 0,
                "slice_count": 0
            }
        
        total_stats[c]["total_src_pixels"] += src_class_pixels
        total_stats[c]["total_enh_pixels"] += enh_class_pixels
        total_stats[c]["total_diff"] += abs(src_class_pixels - enh_class_pixels)
        total_stats[c]["slice_count"] += 1

    return total_stats


def save_statistics(total_stats, csv_path):
    """
    Сохраняет накопленную статистику в CSV-файл с добавлением процентов и долей.

    Parameters:
        total_stats (dict): Накопленная статистика.
        csv_path (str): Путь к выходному CSV-файлу.
    """
    results = []

    # Calculate the total pixels across all slices for normalization
    total_pixels_src = sum([stat["total_src_pixels"] for stat in total_stats.values()])
    total_pixels_enh = sum([stat["total_enh_pixels"] for stat in total_stats.values()])

    for class_id, stat in total_stats.items():
        mean_src_pixels = stat["total_src_pixels"] / stat["slice_count"]
        mean_enh_pixels = stat["total_enh_pixels"] / stat["slice_count"]
        mean_diff_per_slice = stat["total_diff"] / stat["slice_count"]

        # Calculate percentages (now fractions instead of percentages)
        src_fraction = (stat["total_src_pixels"] / total_pixels_src) if total_pixels_src > 0 else 0
        enh_fraction = (stat["total_enh_pixels"] / total_pixels_enh) if total_pixels_enh > 0 else 0

        results.append([class_id,
                        stat["total_src_pixels"],
                        stat["total_enh_pixels"],
                        round(mean_src_pixels, 3),  # Среднее пикселей на срез (исходная разметка)
                        round(mean_enh_pixels, 3),  # Среднее пикселей на срез (исправленная разметка)
                        stat["total_diff"],  # Общая разница, пкс
                        round(mean_diff_per_slice, 3),  # Разница на срез
                        round(src_fraction, 3),  # Доля пикселей для исходной разметки
                        round(enh_fraction, 3)])  # Доля пикселей для улучшенной разметки

    # Проверяем, существует ли файл
    file_exists = os.path.isfile(csv_path)

    # Открываем файл для записи
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Записываем заголовок только один раз
        if not file_exists:
            writer.writerow(["Класс", 
                             "Общее количество пикселей в исходной разметке", 
                             "Общее количество пикселей в улучшенной разметке", 
                             "Среднее количество пикселей на срез в исходной разметке", 
                             "Среднее количество пикселей на срез в исправленной разметке",
                             "Общая разница, пкс", 
                             "Разница на срез",
                             "Доля пикселей в исходной разметке",
                             "Доля пикселей в улучшенной разметке"])

        # Записываем статистику
        writer.writerows(results)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_masks_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/layers_masks_src'  # Specify the directory with original masks
    enh_masks_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/layers_data/layers_masks_enh'  # Specify the directory with enhanced masks
    csv_path = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/class_stats.csv'  # Path for saving CSV statistics

    os.makedirs("pred", exist_ok=True)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("mistakes", exist_ok=True)

    masks_filenames = sorted(os.listdir(enh_masks_dir), key=extract_number)

    total_stats = {}

    for filename in masks_filenames:
        src_mask_path = os.path.join(src_masks_dir, filename)
        enh_mask_path = os.path.join(enh_masks_dir, filename)
        src_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
        src_mask = preprocess_mask(src_mask, config)

        enh_mask = cv2.imread(enh_mask_path, cv2.IMREAD_GRAYSCALE)
        enh_mask = preprocess_mask(enh_mask, config)

        # Calculate statistics for both annotations and accumulate across slices
        total_stats = calculate_statistics(src_mask, enh_mask, config, total_stats)

    # Save accumulated statistics to CSV
    save_statistics(total_stats, csv_path)


if __name__ == "__main__":
    main()
