import torch
import os
import sys
import torch.nn.functional as F
import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from unet import Unet
from residual_unet import ResidualUnet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResidualUnet(1, 6).to(device)

checkpoint = torch.load(
    '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments/20250129-0034/20250129-0034_0300.pth',
    map_location=device
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

activations = None
gradients = None

def save_activations(module, input, output):
    global activations
    activations = output
    activations.retain_grad()  

def save_gradients(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]  

target_layer = model.outputs  
target_layer.register_forward_hook(save_activations)
target_layer.register_backward_hook(save_gradients)

input_img = tifffile.imread('/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/data_layers_seq/train/x/reco_160.tif')

input_tensor = torch.tensor(input_img, dtype=torch.float32)
if input_tensor.ndim == 2:
    input_tensor = input_tensor.unsqueeze(0)  
input_tensor = input_tensor.unsqueeze(0).to(device) 

input_tensor.requires_grad = True

output = model(input_tensor) 
output_softmax = F.softmax(output, dim=1)  

for predicted_class in range(6):
    pred_mask = (output.argmax(dim=1) == predicted_class)  

    target_logits = output[:, predicted_class, :, :][pred_mask]

    if target_logits.numel() == 0:
        continue

    selected_pixels = np.random.choice(target_logits.numel(), size=100, replace=False)
    cam_maps = [] 
    
    for pixel_idx in tqdm(selected_pixels, desc=f'Class {predicted_class}'):
        model.zero_grad()
        target_loss = target_logits[pixel_idx]
        target_loss.backward(retain_graph=True)

        # if gradients is None or activations is None:
        #     print(f"No gradients for class {predicted_class}")
        #     continue

        pooled_gradients = gradients.mean(dim=(2, 3), keepdim=True)

        cam = F.relu((activations * pooled_gradients).sum(dim=1)).detach().cpu().numpy()
        cam_maps.append(cam[0])

    cam_total = np.mean(cam_maps, axis=0)
    cam_total = ((cam_total - cam_total.min()) / (cam_total.max() - cam_total.min())) * 255
    cam_total = cam_total.astype(np.uint8)

    input_image = input_tensor[0, 0].detach().cpu().numpy()
    if input_image.dtype != np.uint8:
        input_image = (input_image * 255).astype(np.uint8)

    # print(input_image.shape)
    # print(cam_total.shape)
    # cv2.resize(cam_total, (input_tensor.shape[1], input_tensor.shape[0]))

    cam_total = cv2.applyColorMap(cam_total, cv2.COLORMAP_JET)
    print(input_image.shape)
    print(cam_total.shape)
    overlayed_img = cv2.addWeighted(np.stack([input_image]*3, axis=-1), 0.7, cam_total, 0.3, 0)

    save_dir = 'grad_cam_res_unet_v2'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"gc_class_{predicted_class}_res_unet2_160.png"
    cv2.imwrite(os.path.join(save_dir, save_path), overlayed_img)
    print(f"Saved to: {save_path}") 


