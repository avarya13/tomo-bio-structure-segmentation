import os
import numpy as np
from PIL import Image
import tifffile
from logging_setup import log_augmentations, initialize_logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2


class OlfBulbDataset(Dataset):
    """
	A PyTorch Dataset for loading images and their corresponding masks.

	Attributes:
		images_paths (list): List of paths to the images.
		masks_paths (list): List of paths to the masks.
		mode (str): Mode of the dataset ('train', 'val', etc.).
		config (object): Configuration object containing settings.
		augmentation (callable): Augmentation function to apply to images and masks.
		save_augmented_dir (str): Directory to save augmented images and masks.
		logger (logging.Logger): Logger for logging augmentation information.
	"""

    def __init__(self, images_dir, masks_dir, config, save_augmented_dir=None, mode='train', val_interval=None, augmentation=None, inference=False):
        """
		Initialize the dataset.

		Parameters:
			images_dir (str): Directory containing input images.
			masks_dir (str): Directory containing mask images.
			config (object): Configuration object.
			save_augmented_dir (str, optional): Directory to save augmented images. Defaults to None.
			mode (str, optional): Mode of the dataset ('train', 'val'). Defaults to 'train'.
			val_interval (int, optional): Validation interval for separating train and validation data. Defaults to None.
			augmentation (callable, optional): Augmentation function. Defaults to None.
		"""

        self.images_paths = [os.path.join(images_dir, id) for id in sorted(os.listdir(images_dir))]
        self.masks_paths = [os.path.join(masks_dir, id) for id in sorted(os.listdir(masks_dir))]
        self.mode = mode
        self.inference = inference

        self.config = config

        """ if not val_interval:
            self.val_interval = config.VAL_INTERVAL
        else:
            self.val_interval = val_interval """

        self.augmentation = augmentation
        self.save_augmented_dir = save_augmented_dir        

        if self.save_augmented_dir:
            os.makedirs(os.path.join(self.save_augmented_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.save_augmented_dir, 'masks'), exist_ok=True)

        """ if self.mode == 'train':
            self.images_paths = [self.images_paths[i] for i in range(len(self.images_paths)) if i % self.val_interval != 0]
            self.masks_paths = [self.masks_paths[i] for i in range(len(self.masks_paths)) if i % self.val_interval != 0]
          
        elif self.mode == 'val':
            self.images_paths = [self.images_paths[i] for i in range(len(self.images_paths)) if i % self.val_interval == 0]
            self.masks_paths = [self.masks_paths[i] for i in range(len(self.masks_paths)) if i % self.val_interval == 0] """
        
        # self.logger = initialize_logging(config)


    def __len__(self):
        """
		Return the total number of samples in the dataset.

		Returns:
			int: Total number of samples (images).
		"""

        return len(self.images_paths)


    def __getitem__(self, index):
        """
		Fetch the sample and its corresponding mask at the specified index.

		Parameters:
			index (int): Index of the sample to fetch.

		Returns:
			tuple: A tuple containing the image tensor, mask tensor, and the image filename.

		Raises:
			IndexError: If the index is out of range.
		"""

        if index >= len(self.images_paths):
            raise IndexError(f"Index {index} is out of range for dataset with length {len(self.images_paths)}")
        
        if len(self.images_paths) != len(self.masks_paths):
            raise IndexError(f"Mismatch between the number of images and masks: {len(self.images_paths)} != {len(self.masks_paths)}")

        image_path = self.images_paths[index]
        mask_path = self.masks_paths[index]

        """ if self.config.INPUT_CHANNELS == 3:
            prev_index = index - 1 
            next_index  = index + 1 if index < len(self.images_paths) - 1 else 0

            prev_image = tifffile.imread(self.images_paths[prev_index])
            cur_image = tifffile.imread(self.images_paths[index])
            next_image = tifffile.imread(self.images_paths[next_index])
            image = np.stack([prev_image, cur_image, next_image], axis=-1)
        else:
            image = tifffile.imread(self.images_paths[index]) """

        image = tifffile.imread(self.images_paths[index])
        mask = np.array(Image.open(self.masks_paths[index]).convert('L'))
        
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # if self.mode == 'train':
            #     log_augmentations(augmented, os.path.basename(image_path), self.mode, self.logger)

        if self.save_augmented_dir and self.config.SAVE_AUG:
            image_save_path = os.path.join(self.save_augmented_dir, 'images', f'{os.path.splitext(os.path.basename(image_path))[0]}.tiff')
            mask_save_path = os.path.join(self.save_augmented_dir, 'masks', f'{os.path.splitext(os.path.basename(mask_path))[0]}.png')

            tifffile.imwrite(image_save_path, image)
            Image.fromarray(mask).save(mask_save_path)

        mask = self.preprocess_mask(mask) 

        transform = transforms.ToTensor()    
        image = transform(image)
        mask = transform(mask)

        return image, mask, os.path.basename(image_path), index
    
    def preprocess_mask(self, mask):
        """
		Preprocess the mask by converting colors to class labels.

		Parameters:
			mask (numpy.ndarray): The mask to preprocess.

		Returns:
			numpy.ndarray: The processed mask with class labels.
		"""
        mask =  mask.astype(np.float32)
        class_colors = self.config.GrayscaleColors
        labels = self.config.SegmentationClass
        
        mask = np.where(mask == class_colors.BACKGROUND_COLOR.value, labels.BACKGROUND.value, mask)
        mask = np.where(mask == class_colors.SL.value, labels.SL.value, mask)
        mask = np.where(mask == class_colors.GL.value, labels.GL.value, mask)
        mask = np.where(mask == class_colors.EPL.value, labels.EPL.value, mask)
        mask = np.where(mask == class_colors.MCL.value, labels.MCL.value, mask)
        mask = np.where(mask == class_colors.GCL.value, labels.GCL.value, mask)

        # if not self.inference and class_colors.IGNORE_INDEX_TRAIN.value:
        #     mask = np.where(mask == class_colors.IGNORE_INDEX_TRAIN.value, labels.IGNORE_INDEX_TRAIN.value, mask)
        # if  self.inference:
        if class_colors.IGNORE_INDEX.value:
            mask = np.where(mask == class_colors.IGNORE_INDEX.value, labels.IGNORE_INDEX.value, mask)
        
        if class_colors.IGNORE_INDEX_2.value:
            mask = np.where(mask == class_colors.IGNORE_INDEX_2.value, labels.IGNORE_INDEX_2.value, mask)

        return mask.astype(np.float32)  

