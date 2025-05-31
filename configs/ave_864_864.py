from pathlib import Path
import random
import torch
from torch.optim import Adam, RMSprop
import numpy as np
import albumentations as A
from enum import Enum

# ==============================
# Paths and Directories
# ==============================
current_dir = Path(__file__).resolve().parent.parent

PROJECT_DIR = current_dir
EXPERIMENTS_DIR = current_dir / 'experiments'
DATA_DIR = current_dir / 'data'

# Data directories
TRAIN_DIR = DATA_DIR / 'train/x'
TRAIN_MASKS_DIR = DATA_DIR / 'train/y'
VAL_DIR = DATA_DIR / 'val/x'
VAL_MASKS_DIR = DATA_DIR / 'val/y'
TEST_DIR = DATA_DIR / 'test/x'
TEST_MASKS_DIR = DATA_DIR / 'test/y'

# Directories for inference
DATA_DIR_INF = current_dir / 'data'
TRAIN_DIR_INF = DATA_DIR_INF / 'train_inf/x'
VAL_DIR_INF = DATA_DIR_INF / 'val/x'
TRAIN_MASKS_DIR_INF = DATA_DIR_INF / 'train_inf/y'
VAL_MASKS_DIR_INF = DATA_DIR_INF / 'val/y'
TEST_DIR_INF = DATA_DIR_INF / 'test/x'
TEST_MASKS_DIR_INF = DATA_DIR_INF / 'test/y'

# Results directories
VAL_RESULTS_DIR = current_dir / 'val_results'
TEST_RESULTS_DIR = current_dir / 'test_results'

# Augmented data directories
TRAIN_AUG_DIR =  current_dir / 'augmented_data/train_aug'
VAL_AUG_DIR = current_dir / 'augmented_data/val_aug'
TEST_AUG_DIR =  current_dir / 'augmented_data/test_aug'

LOSS_WEIGHTS_PATH = 'weights/weights_864_864.tif' 

# ==============================
# General Experiment Settings
# ==============================
FINE_TUNE = False
TASK_TYPE = 'layer segmentation'

# ==============================
# Loss and Optimization
# ==============================
LOSS_MODE = 'multiclass'
CRITERION = 'focal'
OPTIMIZER = Adam

# RMSprop parameters (if used)
WEIGHT_DECAY = 1e-8
MOMENTUM = 0.9

# Learning rate settings
LR = 0.5 * 1e-3
LR_FINE_TUNE = 1e-3
SCHEDULER = None

# ==============================
# Segmentation Metrics
# ==============================
REDUCTION_TYPES = ['macro', 'micro', 'micro-imagewise'] 
SEGM_METRICS = ['iou', 'accuracy', 'precision', 'recall', 'f1']
OBJECT_METRICS = ['precision', 'recall']

# ==============================
# Image and ROI Settings
# ==============================
ROI_HEIGHT = 864
ROI_WIDTH = 864

INPUT_HEIGHT = 864  
INPUT_WIDTH = 864   
INPUT_CHANNELS = 1

# ==============================
# Training Hyperparameters
# ==============================
BATCH_SIZE = 2
EPOCHS = 1000
CH_AGGR_TYPE = 'conv'
VAL_INTERVAL = 12
SAVE_INTERVAL = 10

# ==============================
# Random Seed for Reproducibility
# ==============================
SEED = 23
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# Data Augmentations
# ==============================
TRAIN_AUGMENTATION = A.ReplayCompose([
    A.Resize(INPUT_HEIGHT, INPUT_WIDTH, p=1.0),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(alpha=20, sigma=10, p=0.2),
    A.Rotate(limit=(-180, 180), p=1.0),
])

TRAIN_INF_AUGMENTATION = A.ReplayCompose([A.Resize(INPUT_HEIGHT, INPUT_WIDTH, p=1.0)])
VAL_AUGMENTATION = A.Resize(INPUT_HEIGHT, INPUT_WIDTH, p=1.0)
TEST_AUGMENTATION = A.Resize(INPUT_HEIGHT, INPUT_WIDTH, p=1.0)

AUG_LOGGING_PATH = current_dir / 'aug_logs'
SAVE_AUG = False 

# ==============================
# Layer Freezing (Fine-Tuning)
# ==============================
LAYERS_TO_FREEZE = ['e1']

# ==============================
# Segmentation Class Labels
# ==============================
class SegmentationClass(Enum):
    BACKGROUND = 0
    SL = 1
    GL = 2
    EPL = 3
    MCL = 4
    GCL = 5
    IGNORE_INDEX = -1    
    IGNORE_INDEX_2 = -1
    IGNORE_INDEX_TRAIN = -1

NUM_CLASSES = 6

# ==============================
# Colors for Visualizations
# ==============================
class GrayscaleColors(Enum):
    BACKGROUND_COLOR = 0
    SL = 69   
    GL = 109
    EPL = 153
    MCL = 204
    GCL = 255
    IGNORE_INDEX = 50
    IGNORE_INDEX_2 = 127
    IGNORE_INDEX_TRAIN = None

class RGBColors(Enum):
    BACKGROUND_COLOR = (0, 0, 0)
    IGNORE_INDEX_COLOR = (128, 128, 128)
    OBJECT_COLOR = (0, 255, 0)

# Plot colors
TRAIN_PLOT_COLOR = 'blue'   
VAL_PLOT_COLOR = 'green'   
TEST_PLOT_COLOR = 'red'     

# ==============================
# Overlay Visualization Settings
# ==============================
PREDICTIONS_COLOR = [0, 0, 255]
TARGETS_COLOR = [0, 255, 0]

PREDICTION_ALPHAS = [1.0, 0.35, 0.55]
MASK_ALPHAS = [1.0, 0.4]
EDGE_ALPHAS = [1.0, 0.5, 0.5, 0.3]
MASK_EDGE_ALPHAS = [1.0, 0.6]

ENABLE_OVERLAY_PREDICTIONS = False
ENABLE_OVERLAY_COLORED_PREDICTIONS = False
ENABLE_OVERLAY_TRUE_MASKS = False
ENABLE_OVERLAY_EDGES = False
ENABLE_OVERLAY_COLORED_EDGES = True
ENABLE_OVERLAY_MASK_EDGES = False
ENABLE_SHOW_MISTAKES = True 

PREDICTION_OVERLAY_DIR = 'overlays'
COLORED_PREDICTION_OVERLAY_DIR = 'colored_overlays'
MASK_OVERLAY_DIR = 'overlayed_masks'
EDGE_OVERLAY_DIR = 'overlayed_edges'
COLORED_EDGE_OVERLAY_DIR = 'overlayed_edges_colored'
MASK_EDGE_OVERLAY_DIR = 'overlayed_masks_edges'
MISTAKES_DIR = 'mistakes'

SMOOTH_FACTOR = 0.2

# ==============================
# ROI Cropping Regions
# ==============================
CROPPING_REGIONS = {
    'top-right': {
        'top_left': (280, 747),
        'bottom_right': (1896, 1899)
    },
    'center': {
        'top_left': (865, 1097),
        'bottom_right': (1413, 1505)
    }
}

# ==============================
# Normalization Parameters
# ==============================
NORM_LOWER_BOUND = 0.0005
NORM_UPPER_BOUND = 0.0017

# ==============================
# Interpolation Parameters
# ==============================
LAYERS_CONFIG = {
    'SIGMA': 15,
    'THRESHOLD_LOW': 0.4,
    'THRESHOLD_HIGH': 0.7,
    'NORMALIZATION_FACTOR': 1.1,
    'KERNEL_SIZE': 25,
    'NUM_CONV_LAYERS': 10,
}

GLOMERULI_CONFIG = {
    'SIGMA': 3,
    'THRESHOLD_LOW': 0.65, 
    'THRESHOLD_HIGH': 0.95,
    'NORMALIZATION_FACTOR': 1.1,
    'KERNEL_SIZE': 9,
    'NUM_CONV_LAYERS': 8,
}

# ==============================
# Slicing Parameters
# ==============================
TOTAL_SLICES = 300

TRAIN_SLICES_LEN = 240
VAL_SLICES_LEN = 30
TEST_SLICES_LEN = 30

TRAIN_SHIFT = TEST_SLICES_LEN + VAL_SLICES_LEN
VAL_SHIFT = TEST_SLICES_LEN

TRAIN_SLICES = {'start': 60, 'end': 300}
VAL_SLICES = {'start': 30, 'end': 60}
TEST_SLICES = {'start': 0, 'end': 30}

# ==============================
# Other Settings
# ==============================
AUTO_COMMIT = False
