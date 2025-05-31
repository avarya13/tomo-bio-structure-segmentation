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
    parser.add_argument("--mid", action="store_true", help="Extract only middle slices for val and test")
    parser.add_argument("--num_train", type=int, default=10, choices=[10, 2, 1], help="Train set slice selection: 1 (only 5), 2 (4 and 7), 10 (all)")
    parser.add_argument("--exclude_tens", type=int, nargs="*", default=[], help="List of excluded tens")
    parser.add_argument("--test_dec", type=int, required=True, help="Test decade (1-30)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    allowed = [4, 5, 6, 7] if args.mid else list(range(1, 11))
    
    train_indices = []
    test_indices = []
    val_indices = []
    
    test_start = (args.test_dec - 1) * 10
    test_indices.extend([test_start + i for i in allowed if test_start + i < 300])
    
    val_left_dec = args.test_dec - 1
    val_right_dec = args.test_dec + 1
    
    if val_left_dec >= 1 and val_left_dec not in args.exclude_tens:
        val_left_start = (val_left_dec - 1) * 10
        val_indices.extend([val_left_start + i for i in allowed if val_left_start + i < 300])
    
    if val_right_dec <= 30 and val_right_dec not in args.exclude_tens:
        val_right_start = (val_right_dec - 1) * 10
        val_indices.extend([val_right_start + i for i in allowed if val_right_start + i < 300])
    
    for dec in range(1, 31):
        if dec in args.exclude_tens or dec == args.test_dec or dec == val_left_dec or dec == val_right_dec:
            continue 
        
        start = (dec - 1) * 10
        
        if args.num_train == 1:
            selected_slices = [5]
        elif args.num_train == 2:
            selected_slices = [4, 7]
        else:
            selected_slices = list(range(1, 11))
        
        train_indices.extend([start + i for i in selected_slices if start + i < 300 and start + i not in val_indices and start + i not in test_indices])
    
    create_lst(train_indices, "reco", output_dir / f"train{args.test_dec:02d}.txt")
    create_lst(val_indices, "reco", output_dir / f"val_{args.test_dec:02d}.txt")
    create_lst(test_indices, "reco", output_dir / f"test_{args.test_dec:02d}.txt")

if __name__ == "__main__":
    main()
