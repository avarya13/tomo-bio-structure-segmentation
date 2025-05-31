import os
import sys
from pathlib import Path
import logging 
from datetime import datetime


def setup_logger(name, log_file, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', to_console=False): 
    """
	Setup a logger for logging messages to a file and optionally to the console.

	Parameters:
		name (str): The name of the logger.
		log_file (str): The file to which logs will be written.
		level (int): The logging level (default: logging.INFO).
		format (str): The format of the log messages (default: '%(asctime)s %(levelname)s %(message)s').
		to_console (bool): Whether to also log messages to the console (default: False).

	Returns:
		logging.Logger: Configured logger instance.
	"""
    formatter = logging.Formatter(format)
    
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_augmentations(augmented, name, mode, aug_logger):
    """
	Log the applied augmentations for a given item.

	Parameters:
		augmented (dict): Dictionary containing augmentation details.
		name (str): Name of the item being augmented.
		mode (str): Mode of the operation (e.g., 'train', 'val').
		aug_logger (logging.Logger): Logger instance for logging augmentations.
	"""

    replay_augmentations = augmented['replay']['transforms']

    applied_augmentations = []
    for aug in replay_augmentations:
        if aug['applied']:
            applied_augmentations.append(aug)

    if applied_augmentations:            
        aug_logger.info(f"Applied augmentations on {name} ({mode}): {applied_augmentations}")


def initialize_logging(config):
    """
	Initialize logging for augmentations.

	Creates the logging directory and sets up the logger for augmentations.

	Parameters:
		config (object): Configuration object containing logging settings.

	Returns:
		logging.Logger: Configured logger instance for augmentations.
	"""
    
    current_datetime = datetime.now().strftime('%Y%m%d-%H%M')

    log_dir = Path(config.AUG_LOGGING_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = log_dir / f"aug_{current_datetime}.log"

    return setup_logger('aug_logger', log_filename)
 



