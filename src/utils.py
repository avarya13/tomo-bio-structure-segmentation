import subprocess
import torch
from pathlib import Path
from collections import namedtuple
from unet import Unet
from unet_small import SmallUnet
from residual_unet import ResidualUnet
from residual_unet_small import ResidualUnetSmall
from unet_ref_article import UNet


def save_checkpoint(model, optimizer, epoch, save_path, epoch_loss_train, epoch_loss_val, logger, scheduler=None):
    """
    Saves the model checkpoint to a specified file.

    Args:
        model (torch.nn.Module): The model whose state to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): Current epoch number.
        save_path (str): Path where the checkpoint will be saved.
        epoch_loss_train (list): Training loss values for the epoch.
        epoch_loss_val (list): Validation loss values for the epoch.
        logger (logging.Logger): Logger for tracking the saving process.
        scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler state to save.

    Raises:
        Exception: If saving the checkpoint fails.
    """

    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': epoch_loss_train,
            'loss_val': epoch_loss_val,
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()  

        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint successfully saved to {save_path}.")

    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def log_layer_states(model, logger):
    """
    Saves the model checkpoint to a specified file.

    Args:
        model (torch.nn.Module): The model whose state to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): Current epoch number.
        save_path (str): Path where the checkpoint will be saved.
        epoch_loss_train (list): Training loss values for the epoch.
        epoch_loss_val (list): Validation loss values for the epoch.
        logger (logging.Logger): Logger for tracking the saving process.

    Raises:
        Exception: If saving the checkpoint fails.
    """

    logger.info("Logging layer states...")
    
    def log_layer(layer, name_prefix=''):
        for name, param in layer.named_parameters():
            if param.requires_grad:
                logger.info(f"{name_prefix}{name} - requires_grad=True")
            else:
                logger.info(f"{name_prefix}{name} - requires_grad=False")
           
        for name, sub_module in layer.named_children():
            log_layer(sub_module, name_prefix + name + '.')

    log_layer(model)


def get_model(model, input_channels, num_classes, channels_aggr, device):
    if model == 'unet_small':
        return SmallUnet(input_channels, num_classes).to(device)
    elif model == 'unet':
        return Unet(input_channels, num_classes).to(device)
    elif model == 'res_unet':
        return ResidualUnet(input_channels, num_classes, channels_aggr).to(device)
    elif model == 'res_unet_small':
        return ResidualUnetSmall(input_channels, num_classes, channels_aggr).to(device)
    elif model == 'unet_ref':
        return UNet(input_channels, num_classes).to(device)
    else:
        raise ValueError(f'Unknown model name: {model}')


def setup_model_and_stats(model, save_dir, device, config, logger, fine_tune=False):
    """
    Sets up the model and loads its state from the latest checkpoint, if available.

    Args:
        model (torch.nn.Module): The model to set up.
        save_dir (str): Directory where checkpoints are saved.
        device (torch.device): Device to load the model onto (CPU or GPU).
        config (Config): Configuration object containing model and optimizer settings.
        logger (logging.Logger): Logger for tracking the loading process.
        fine_tune (bool, optional): If True, enter fine-tuning mode. Default is False.

    Returns:
        LoadedData: A namedtuple containing the loaded model, optimizer, epoch, and loss statistics.
    
    Raises:
        FileNotFoundError: If fine-tuning is selected but no checkpoint is found.
    """

    model = get_model(model, config.INPUT_CHANNELS, config.NUM_CLASSES, config.CH_AGGR_TYPE, device)
    model_name = model.__class__.__name__  
    logger.info(f"Model: {model_name}")

    loaded_data = namedtuple('LoadedData', [
        'model',
        'optimizer',
        'scheduler',
        'start_epoch',
        'loss_train',
        'loss_val',
    ])

    checkpoint_files = list(Path(save_dir).glob("*.pth"))

    if fine_tune and not checkpoint_files:
        logger.error("Fine-tuning mode selected, but no checkpoint found. Please, train the model first without 'fine tuning'.")
        raise FileNotFoundError("No checkpoint found for fine-tuning.")

    optimizer_lr = config.LR_FINE_TUNE if fine_tune else config.LR
    optimizer = config.OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer_lr)
    
    if config.SCHEDULER:
        scheduler = config.SCHEDULER(optimizer, factor=0.1, mode='min', threshold=1e-4, threshold_mode='abs', eps=1e-8)
        # scheduler = config.SCHEDULER(optimizer, config.STEP_SIZE, config.GAMMA)
        # scheduler = config.SCHEDULER(optimizer, mode='min', patience=2, verbose=True, threshold=1e-4)
    else:
        scheduler = None

    if config.TASK_TYPE:
        logger.info(f'Starting training mode for {config.TASK_TYPE}.')
    else:
        logger.info(f'Starting training mode.')

    if not checkpoint_files:
        logger.info("No checkpoint found. Starting from scratch.")
        return loaded_data(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=0,
            loss_train=[],
            loss_val=[],
        )

    latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
    logger.info(f"Loading model from latest checkpoint: {latest_checkpoint}")

    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)

    start_epoch = checkpoint['epoch']
    loss_train = checkpoint.get('loss_train', [])
    loss_val = checkpoint.get('loss_val', [])

    if fine_tune:
        logger.info("Fine-tuning mode: freezing layers.")
        adjust_layers(model, config.LAYERS_TO_FREEZE, logger)
        optimizer_lr = config.LR_FINE_TUNE
        logger.info("Fine-tuning mode: optimizer initialized with unfrozen layers only.")
    else:
        optimizer_lr = config.LR
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state successfully loaded from checkpoint.")
        else:
            logger.info("No optimizer state found in checkpoint, using new optimizer.")

    optimizer = config.OPTIMIZER(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer_lr)

    logger.info(f"Model and optimizer state successfully loaded from {latest_checkpoint}.")
    #log_layer_states(model, logger)

    return loaded_data(
        model=model.to(device),
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        loss_train=loss_train,
        loss_val=loss_val,
    )


def adjust_layers(model, layers_to_freeze, logger):
    """
    Adjusts the layers of the model to set their requires_grad attribute based on the provided layers to freeze.

    Args:
        model (torch.nn.Module): The model to adjust layers for.
        layers_to_freeze (list): List of layer names to freeze.
        logger (logging.Logger): Logger for tracking the adjustment process.
    """

    for param in model.parameters():
        param.requires_grad = True
    
    for layer_name in layers_to_freeze:
        layer = getattr(model, layer_name, None)
        if layer:
            for param_name, param in layer.named_parameters():
                param.requires_grad = False
                logger.info(f"Layer {layer_name}: Parameter {param_name} set to requires_grad=False")


def run_command(command, logger):
    """
    Executes a shell command and logs the output.

    Args:
        command (list): Command to be executed as a list of strings.
        logger (logging.Logger): Logger for tracking command execution.

    Returns:
        str: Standard output from the command execution, or None if an error occurs.
    """
    
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            logger.error(f'Command "{command}" failed with return code {process.returncode}. Error: {stderr}')

        return stdout

    except Exception as ex:
        logger.error(f'An exception occurred while executing the command "{command}": {ex}')
        return None


def commit_changes(comment, logger, push=False):
    """
    Commits changes to the current Git branch and optionally pushes them to the remote repository.

    Args:
        comment (str): Commit message describing the changes.
        logger (logging.Logger): Logger for tracking the commit process.
        push (bool, optional): If True, pushes the commit to the remote repository. Default is False.

    Returns:
        None
    """

    branch = run_command(['git', 'branch', '--show-current'], logger).strip()
    
    if not branch:
        logger.warning("Failed to determine the current Git branch")
        return

    run_command(['git', 'add', '.'], logger)
    run_command(['git', 'commit', '-m', comment], logger)

    if push:
        run_command(['git', 'pull', 'origin', branch], logger)
        run_command(['git', 'push', 'origin', branch], logger)

    logger.info(f"Changes committed to branch {branch}")
    if push:
        logger.info("Changes pushed to remote repository")
