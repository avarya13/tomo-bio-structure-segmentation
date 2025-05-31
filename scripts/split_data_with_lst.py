import os
import shutil
import argparse
from pathlib import Path

def read_file_lst(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def copy_files(file_lst, src_dir, dst_dir, subfolders):
    for file in file_lst:
        for subfolder in subfolders:
            ext = ".tif" if subfolder == "x" else ".png"
            src_path = src_dir / subfolder / f"{file}{ext}"
            
            if src_path.exists():
                dst_path = dst_dir / subfolder / f"{file}{ext}"
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"From {src_path} to {dst_path}")
                shutil.copy2(src_path, dst_path)

def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test data")
    parser.add_argument("--data_dir", required=True, help="Path to the source folder")
    parser.add_argument("--train_lst", required=True, help="File containing the lst of images for the train data")
    parser.add_argument("--val_lst", required=True, help="File containing the lst of images for the validation data")
    parser.add_argument("--test_lst", required=True, help="File containing the lst of images for the test data")
    parser.add_argument("--output_dir", required=True, help="Directory where the output will be saved")

    args = parser.parse_args()

    src_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    categories = ["train", "val", "test"]
    subfolders = ["x", "y"]
    file_lsts = {
        "train": read_file_lst(args.train_lst),
        "val": read_file_lst(args.val_lst),
        "test": read_file_lst(args.test_lst),
    }
    print(file_lsts)

    for category in categories:
        dst_dir = output_dir / category
        copy_files(file_lsts[category], src_dir, dst_dir, subfolders)

if __name__ == "__main__":
    main()
