import tifffile

mask = tifffile.imread('/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/weights/weights_x2.tif')  # shape: (num_slices, height, width)

mid = mask.shape[2] // 2

left_half = mask[:, :, :mid]
right_half = mask[:, :, mid:]

tifffile.imwrite('weights/dt_left.tif', left_half)
tifffile.imwrite('weights/dt_right.tif', right_half)
