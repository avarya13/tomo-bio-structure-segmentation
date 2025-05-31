import cv2
import numpy as np
import tifffile

num1=tifffile.imread('/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/ens_res/_num1.tif')
num2=tifffile.imread('/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/ens_res/_num2.tif')

confids=np.zeros((300,288*2,800),np.float32)



for i_row in range(0,288*2):
    for i_col in range(0,800):
        for i_sla in range(0,300):
            n1=num1[i_sla,i_row,i_col]
            if n1 < 30:
                n2=num2[i_sla,i_row,i_col]
                confids[i_sla,i_row,i_col]=(1.0*n1*(n1-n2))/(30.0*29.0)
            else:
                confids[i_sla,i_row,i_col]=1.0

tifffile.imwrite('/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/ens_res/_conf.tif', confids, compression="zlib")

