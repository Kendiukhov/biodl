import os
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any
import logging
import time

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """
    Saves configuration parameters to a YAML file.

    Args:
        config: Configuration dictionary.
        save_path: Path to save the configuration file.
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f)
    logging.info(f'Configuration saved to {save_path}.')

def setup_logging(log_file: str = None, log_level: int = logging.INFO):
    """
    Sets up logging for the application.

    Args:
        log_file: Path to a file to log to (optional).
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(level=log_level, format=log_format, filename=log_file, filemode='w')
    else:
        logging.basicConfig(level=log_level, format=log_format)

    # Optionally, add logging to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_random_seed(seed: int):
    """
    Sets the random seed for reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f'Random seed set to {seed}.')

def get_device(use_gpu: bool = True) -> torch.device:
    """
    Determines the device to run computations on.

    Args:
        use_gpu: Whether to use GPU if available.

    Returns:
        A torch.device object.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info('Using GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device('cpu')
        logging.info('Using CPU.')
    return device

class Timer:
    """
    A simple timer context manager for measuring code execution time.
    """
    def __init__(self, name: str = None):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        if self.name:
            logging.info(f'[{self.name}] Timer started.')

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time
        if self.name:
            logging.info(f'[{self.name}] Timer stopped. Elapsed time: {elapsed_time:.2f} seconds.')
        else:
            logging.info(f'Timer stopped. Elapsed time: {elapsed_time:.2f} seconds.')

def count_parameters(model: torch.nn.Module) -> int:
    """
    Counts the number of trainable parameters in a model.

    Args:
        model: The model to inspect.

    Returns:
        The total number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Model has {total_params} trainable parameters.')
    return total_params

def save_model(model: torch.nn.Module, save_path: str):
    """
    Saves the model's state dictionary.

    Args:
        model: The model to save.
        save_path: Path to save the model.
    """
    torch.save(model.state_dict(), save_path)
    logging.info(f'Model saved to {save_path}.')

def load_model(model: torch.nn.Module, load_path: str, device: torch.device):
    """
    Loads the model's state dictionary.

    Args:
        model: The model to load the state into.
        load_path: Path to the saved state dictionary.
        device: Device to map the model to.

    Returns:
        The model with loaded state dictionary.
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    logging.info(f'Model loaded from {load_path}.')
    return model
