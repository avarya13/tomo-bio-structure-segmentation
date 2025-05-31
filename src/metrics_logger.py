import pandas as pd
import os
import csv

class MetricLogger:
    """
    A class to log and save metrics during model training and validation.

    This class handles the creation of metric logs, including the ability to save
    metrics to a CSV file and log summaries of each epoch.

    Attributes:
        mode (str): The current mode ('train' or 'val').
        metrics_dir (str): The directory where metrics will be saved.
        config (object): Configuration object containing metric settings.
        timestamp (str): Timestamp for the current logging session.
        metrics_df (pd.DataFrame): DataFrame holding the metrics data.
    """

    def __init__(self, mode, metrics_dir, config, timestamp):
        """
        Initialize the MetricLogger.

        Parameters:
            mode (str): The mode of operation ('train' or 'val').
            metrics_dir (str): The path to the CSV file for saving metrics.
            config (object): Configuration object containing metric settings.
            timestamp (str): Timestamp for naming the model during logging.

        Raises:
            FileNotFoundError: If the metrics directory does not exist and cannot be created.
        """

        self.mode = mode
        self.metrics_dir = metrics_dir
        self.config = config
        self.timestamp = timestamp
        
        if os.path.exists(self.metrics_dir):
            self.metrics_df = pd.read_csv(self.metrics_dir)
        else:
            self.metrics_df = pd.DataFrame(columns=['model'] + self.create_header())
            self.metrics_df.to_csv(self.metrics_dir, index=False)  

    def create_header(self):
        """
        Create the header for the metrics CSV file.

        This method generates the column headers based on the configured segmentation metrics
        and reduction types.

        Returns:
            list: A list of header strings for the CSV file.
        """
        
        header = []
        for metric in self.config.SEGM_METRICS:
            for reduction in self.config.REDUCTION_TYPES:
                if reduction != 'object':
                    header.append(f'{metric}_{reduction}')
        for metric in self.config.OBJECT_METRICS:
            header.append(f'{metric}_object')
        return header

    def get_model_name(self, epoch):
        """
        Generate the model name for a given epoch.

        The model name is formatted to include a timestamp and zero-padded epoch number.

        Parameters:
            epoch (int): The current epoch number.

        Returns:
            str: The formatted model name.
        """
        return f'{self.timestamp}_{str(epoch).zfill(4)}'

    def save_epoch_metrics(self, logger, epoch, total_epochs, metrics, loss):
        """
        Save the metrics for the current epoch to the CSV file.

        This method flattens the metrics dictionary and appends the results to the CSV file.
        It also logs a summary of the epoch's performance.

        Parameters:
            logger (logging.Logger): Logger instance for recording summary information.
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs for training.
            metrics (dict): Dictionary containing computed metrics for the epoch.
            loss (float): The loss value for the epoch.
        """
        model_name = self.get_model_name(epoch)
        metrics_flat = {'model': model_name}
        metrics_flat.update(metrics)

        file_exists = os.path.exists(self.metrics_dir)

        with open(self.metrics_dir, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                writer.writerow(['model'] + self.create_header())  

            writer.writerow(metrics_flat.values())

        self.log_epoch_summary(logger, epoch, total_epochs, metrics, loss)


    def log_epoch_summary(self, logger, epoch, total_epochs, metrics, loss):  
        """
        Log a summary of the epoch's performance.

        This method logs the epoch number, phase (training or validation), loss, and metrics.

        Parameters:
            logger (logging.Logger): Logger instance for recording information.
            epoch (int): The current epoch number.
            total_epochs (int): The total number of epochs for training.
            metrics (dict): Dictionary containing computed metrics for the epoch.
            loss (float): The loss value for the epoch.

        Raises:
            ValueError: If the mode is not recognized.
        """
        if  self.mode == 'train':
            phase = 'Training' 
        elif self.mode == 'val':
            phase ='Validation'
        else:
            logger.error(f"Unknown mode: {self.mode}. Acceptable options: train, val.")
            return
        
        logger.info(f"Epoch {epoch}/{total_epochs} - {phase} phase...")
        logger.info(f"Epoch {epoch} - {phase} Loss: {loss:.4f}")
        
        for metric in metrics.keys():
                metrics_parts = metric.split('_')
                metric_name = metrics_parts[0]
                reduction = '-'.join(metrics_parts[1:])

                if metric.endswith('object'):
                    logger.info(f"Epoch {epoch} - {phase} Obj_{metric_name.title()}: {metrics[metric]:.4f}")
                else:
                    logger.info(f"Epoch {epoch} - {phase} Pix_{metric_name.title() } ({reduction.replace('_', '-').title()}): {metrics[metric]:.4f}")