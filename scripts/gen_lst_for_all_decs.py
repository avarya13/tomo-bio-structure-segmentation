import os
import argparse
from pathlib import Path

def create_lst(indices, prefix, file_path, mode="w"):
    with open(file_path, mode) as f:
        for i in indices:
            f.writelines(f"{prefix}_{str(i).zfill(4)}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test data")
    parser.add_argument("--output_dir", required=True, help="Directory where the output will be saved")
    parser.add_argument('--mid', action='store_true', help="A flag to extract only middle slices")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_file = output_dir / "train.txt"
    val_file = output_dir / "val.txt"
    test_file = output_dir / "test.txt"

    tens = list(range(0, 300, 10))

    if args.mid:
        allowed = [4, 7]  
    else:
        allowed = list(range(1, 11))

    test_indices = [i for i in range(1, 31)]
    val_indices = [i for i in range(31, 61)]
    train_indices = [i for i in range(61, 300) if i % 10 in allowed]
    
    create_lst(test_indices, "reco", test_file)
    create_lst(val_indices, "reco", val_file)
    create_lst(train_indices, "reco", train_file)

if __name__ == "__main__":
    main()
