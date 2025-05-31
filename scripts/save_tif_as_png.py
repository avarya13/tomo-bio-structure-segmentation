import os
import sys
import argparse
import numpy as np
import tifffile
import importlib
from PIL import Image  

data_dir = r'/home/vkaryakina/OB/examples/roi_tiff'
save_dir = r'/home/vkaryakina/OB/examples/norm_roi'

os.makedirs(save_dir, exist_ok=True)

files = sorted(os.listdir(data_dir))

for index, file in enumerate(files):
    file_path = os.path.join(data_dir, file)
    img = tifffile.imread(file_path)
    print(np.unique((img * 255).astype(np.uint8)))
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img = Image.fromarray((img * 255).astype(np.uint8))
    img.save(os.path.join(save_dir, f'{filename}.png'))
    
