import os
from PIL import Image



def main():
    data_dir = r'E:\cellpose\test\test'
    masks_dir = r'D:\datasets\cellpose_exp\cell\test\y'
    img_dir = r'D:\datasets\cellpose_exp\cell\test\x'

    for file in sorted(os.listdir(data_dir)):
        image = Image.open(os.path.join(data_dir, file))
        if file.endswith('_img.png'):
            image.save(os.path.join(img_dir, f'{file}'))
        elif file.endswith('_masks.png'):
            image.save(os.path.join(masks_dir, f'{file}'))



if __name__ == "__main__":
    main()