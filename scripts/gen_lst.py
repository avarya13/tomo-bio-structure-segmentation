import os
import argparse
from pathlib import Path


def create_lst(indices, prefix, file_path):
    with open(file_path, "w") as f:
        for i in indices:
            f.writelines(f"{prefix}_{str(i).zfill(4)}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test data")
    parser.add_argument("--output_dir", required=True, help="Directory where the output will be saved")
    parser.add_argument('--mid', action='store_true', help="A flag to extract only middle slices")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tens = list(range(0, 300, 10))

    for indx, test_start in enumerate(tens):
        if args.mid:
            allowed = [4, 7]  
        else:
            allowed = list(range(1, 11)) 

        test_indices = [test_start + i for i in allowed if test_start + i < 300]

        val_left = (test_start - 10) % 300
        val_right = (test_start + 10) % 300

        val_indices = [val_left + i for i in allowed if val_left + i < 300] + \
                      [val_right + i for i in allowed if val_right + i < 300]

        all_indices = [i for i in range(1, 300) if i % 10 in allowed]
        train_indices = [i for i in all_indices if i not in test_indices and i not in val_indices]

        dec_num = str(test_start // 10 + 1).zfill(2)

        create_lst(train_indices, "reco", output_dir / f"train_{dec_num}.txt")
        create_lst(val_indices, "reco", output_dir / f"val_{dec_num}.txt")
        create_lst(test_indices, "reco", output_dir / f"test_{dec_num}.txt")


if __name__ == "__main__":
    main()
