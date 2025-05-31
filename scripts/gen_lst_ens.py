import os
import argparse
from pathlib import Path

def create_lst(indices, prefix, file_path):
    with open(file_path, "w") as f:
        for i in indices:
            f.write(f"{prefix}_{str(i).zfill(4)}\n")

def get_slices_in_tens(start, end, current_ten, train_val_slices, test_slices):
    train_val = []
    test = []
    for i in range(start, end, 10):  
        if i == current_ten:
            train_val.extend([i + s for s in train_val_slices])
        else:
            test.extend([i + s for s in test_slices])
    return train_val, test

def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test data")
    parser.add_argument("--output_dir", required=True, help="Directory where the output will be saved")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_slices = list(range(0, 300))  
    train_val_slices = [4, 7]  
    test_slices = [4, 7]  

    for current_ten in range(0, 300, 10): 
        train_val, test = get_slices_in_tens(0, 300, current_ten, train_val_slices, test_slices)
        cur_dec = str(current_ten//10 + 1).zfill(2)
        
        prefix = f"reco"
        create_lst(train_val, prefix, output_dir / f"train_{cur_dec}.txt")
        create_lst(train_val, prefix, output_dir / f"val_{cur_dec}.txt")
        create_lst(test, prefix, output_dir / f"test_{cur_dec}.txt")

    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
