import os
from pathlib import Path


current_dir = Path(__file__).resolve().parent.parent


def get_inference_dirs(mode, config):
    paths = {
        'train': (config.TRAIN_DIR_INF, config.TRAIN_MASKS_DIR_INF, config.TRAIN_AUG_DIR),
        'val': (config.VAL_DIR_INF, config.VAL_MASKS_DIR_INF, config.VAL_AUG_DIR),
        'test': (config.TEST_DIR_INF, config.TEST_MASKS_DIR_INF, config.TEST_AUG_DIR),
    }
    return paths.get(mode, (None, None, None))


def get_augmentation(mode, config):
    augmentations = {
        'train': config.TRAIN_INF_AUGMENTATION,
        'val': config.VAL_AUGMENTATION,
        'test': config.TEST_AUGMENTATION,
    }
    return augmentations.get(mode, None)


def get_image_paths(results_dir, subdir, logger):
    subdir_path = os.path.join(results_dir, subdir)
    if not os.path.exists(subdir_path):
        logger.warning(f"Directory {subdir_path} does not exist.")
    
    file_names = sorted(os.listdir(subdir_path))
    file_paths = [os.path.join(subdir_path, file_name) for file_name in file_names if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    return file_paths


def get_model_path(model_dir, logger, epoch=None):
    checkpoint_files = list(Path(model_dir).glob("*.pth"))

    if not checkpoint_files:
        logger.warning("No model checkpoints found in the directory.")
        return None, None
    
    if epoch is not None:
        epoch = int(epoch)
        model_path = Path(model_dir) / f"{Path(model_dir).name}_{str(epoch).zfill(4)}.pth"
        print('model_path', model_path)
        if model_path in checkpoint_files:
            return model_path, epoch
        else:
            logger.warning(f"No model checkpoint found for epoch {epoch}.")
            return None, None
    
    latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
    latest_epoch = int(latest_checkpoint.stem.split('_')[-1])
    return latest_checkpoint, latest_epoch

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_visualization_directories(results_dir):
    visualization_dir = os.path.join(results_dir)
    ensure_directory_exists(visualization_dir)
    return visualization_dir


