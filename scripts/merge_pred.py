import cv2
import numpy as np
import tifffile
import os

classes=np.zeros((300,288*2,200*2),np.uint8)
cla1=np.zeros((300,288*2,200*2),np.uint8)
num1=np.zeros((300,288*2,200*2),np.uint8)
cla2=np.zeros((300,288*2,200*2),np.uint8)
num2=np.zeros((300,288*2,200*2),np.uint8)

confids=np.zeros((300,288*2,200*2),np.float32)

exp_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.d2.left"
save_dir ='w5s0_left'
os.makedirs(save_dir, exist_ok=True)

timestamps = [tm for  tm in os.listdir(exp_dir)]

for i_s in range(1,301):
    vol=np.zeros((30,288*2,200*2),np.uint8)
    i_r = 0
    for i_dec in range(1,31):
        # path = os.path.join(exp_dir, timestamps[i_dec], f"{timestamps[i_dec]}_2750", "predictions", f"reco_{1:04d}.png")
        img_path='/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.d2.left/00003000-{0:04d}/00003000-{0:04d}_1250/predictions/reco_{1:04d}.png'.format(i_dec,i_s)
        # path = os.path.join(exp_dir, img_path)
        # print(path)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
       
        vol[i_r]=image
        i_r += 1
    vol=np.sort(vol, axis=0)
    img_med=vol[15]
    tifffile.imwrite('{}/sorted.{:02d}.tif'.format(save_dir, i_s), vol, compression="zlib")
    for i_row in range(0,288*2):
        for i_col in range(0,200*2):
            # if i_row < 160:
            #     continue
            # if i_col < 200:
            #     continue
            v=vol[:,i_row,i_col]
            unique, counts = np.unique(v, return_counts=True)
            cnts = np.rec.fromarrays([counts, unique])
            #print(cnts)
            cnts.sort()
            #print(cnts)
            num1[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-1][0]
            cla1[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-1][1]
            if 1 < cnts.shape[0]:
                num2[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-2][0]
                cla2[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-2][1]
            i_cnt = np.argmax(counts)
            idx_class = unique[i_cnt]
            classes[i_s-1,i_row,i_col] = idx_class
            confid = np.count_nonzero(v == classes[i_s-1,i_row,i_col]) / 30.0
            confids[i_s-1,i_row,i_col] = confid
            # if confid < 0.35:
            #     print(unique, counts)
    #tifffile.imwrite('_classes.{:02d}.tif'.format(i_dec), classes, compression="zlib")
    cv2.imwrite('{}/{:02d}.pmed.png'.format(save_dir,i_s),img_med)
    print('slice {:02d} finished'.format(i_s))
tifffile.imwrite(f'{save_dir}/_classes.tif', classes, compression="zlib")
tifffile.imwrite(f'{save_dir}/_confids.tif', confids, compression="zlib")
tifffile.imwrite(f'{save_dir}/_cla1.tif', cla1, compression="zlib")
tifffile.imwrite(f'{save_dir}/_cla2.tif', cla2, compression="zlib")
tifffile.imwrite(f'{save_dir}/_num1.tif', num1, compression="zlib")
tifffile.imwrite(f'{save_dir}/_num2.tif', num2, compression="zlib")



classes=np.zeros((300,288*2,200*2),np.uint8)
cla1=np.zeros((300,288*2,200*2),np.uint8)
num1=np.zeros((300,288*2,200*2),np.uint8)
cla2=np.zeros((300,288*2,200*2),np.uint8)
num2=np.zeros((300,288*2,200*2),np.uint8)

confids=np.zeros((300,288*2,200*2),np.float32)

exp_dir = "/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.d2.left"
save_dir = '/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/consistency_maps'
os.makedirs(save_dir, exist_ok=True)

timestamps = [tm for  tm in os.listdir(exp_dir)]

for i_s in range(1,301):
    vol=np.zeros((30,288*2,200*2),np.uint8)
    i_r = 0
    for i_dec in range(1,31):
        # path = os.path.join(exp_dir, timestamps[i_dec], f"{timestamps[i_dec]}_2750", "predictions", f"reco_{1:04d}.png")
        img_path='/home/vkaryakina/OB/unet-olfactory-bulb-segmentation/experiments.d2.left/00003000-{0:04d}/00003000-{0:04d}_1250/predictions/reco_{1:04d}.png'.format(i_dec,i_s)
        # path = os.path.join(exp_dir, img_path)
        # print(path)
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
       
        vol[i_r]=image
        i_r += 1
    vol=np.sort(vol, axis=0)
    img_med=vol[15]
    tifffile.imwrite('{}/sorted.{:02d}.tif'.format(save_dir, i_s), vol, compression="zlib")
    for i_row in range(0,288*2):
        for i_col in range(0,200*2):
            # if i_row < 160:
            #     continue
            # if i_col < 200:
            #     continue
            v=vol[:,i_row,i_col]
            unique, counts = np.unique(v, return_counts=True)
            cnts = np.rec.fromarrays([counts, unique])
            #print(cnts)
            cnts.sort()
            #print(cnts)
            num1[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-1][0]
            cla1[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-1][1]
            if 1 < cnts.shape[0]:
                num2[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-2][0]
                cla2[i_s-1,i_row,i_col] = cnts[cnts.shape[0]-2][1]
            i_cnt = np.argmax(counts)
            idx_class = unique[i_cnt]
            classes[i_s-1,i_row,i_col] = idx_class
            confid = np.count_nonzero(v == classes[i_s-1,i_row,i_col]) / 30.0
            confids[i_s-1,i_row,i_col] = confid
            # if confid < 0.35:
            #     print(unique, counts)
    #tifffile.imwrite('_classes.{:02d}.tif'.format(i_dec), classes, compression="zlib")
    cv2.imwrite('{}/{:02d}.pmed.png'.format(save_dir,i_s),img_med)
    print('slice {:02d} finished'.format(i_s))
tifffile.imwrite(f'{save_dir}/_classes.tif', classes, compression="zlib")
tifffile.imwrite(f'{save_dir}/_confids.tif', confids, compression="zlib")
tifffile.imwrite(f'{save_dir}/_cla1.tif', cla1, compression="zlib")
tifffile.imwrite(f'{save_dir}/_cla2.tif', cla2, compression="zlib")
tifffile.imwrite(f'{save_dir}/_num1.tif', num1, compression="zlib")
tifffile.imwrite(f'{save_dir}/_num2.tif', num2, compression="zlib")
