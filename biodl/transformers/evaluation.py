import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates the model on a dataset.

    Args:
        model: The trained model to evaluate.
        data_loader: DataLoader for the dataset to evaluate on.
        criterion: Loss function.
        device: Device to perform computation on.

    Returns:
        Tuple containing average loss, perplexity, and accuracy.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for input_ids, target_ids in data_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            output = model(input_ids)

            output_flat = output.view(-1, output.size(-1))
            target_flat = target_ids.view(-1)

            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = output_flat.argmax(dim=1)
            correct = (predictions == target_flat).sum().item()
            total_correct += correct
            total_count += target_flat.size(0)

    avg_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_count

    return avg_loss, perplexity, accuracy

def plot_training_curves(
    training_losses: List[float],
    validation_losses: List[float],
    save_path: str = None
):
    """
    Plots training and validation loss curves.

    Args:
        training_losses: List of training losses per epoch.
        validation_losses: List of validation losses per epoch.
        save_path: Path to save the plot image (optional).
    """
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_losses, 'b-', label='Training Loss')
    plt.plot(epochs, validation_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f'Training curves saved to {save_path}.')
    else:
        plt.show()

def fit_scaling_law(
    x_data: np.ndarray,
    y_data: np.ndarray,
    law_type: str = 'power_law'
) -> Dict[str, Any]:
    """
    Fits a scaling law to the data.

    Args:
        x_data: Independent variable data (e.g., model size, dataset size).
        y_data: Dependent variable data (e.g., loss).
        law_type: Type of scaling law ('power_law' or 'logarithmic').

    Returns:
        Dictionary containing the fit parameters and the fit function.
    """
    if law_type == 'power_law':
        # Power law: y = a * x^b + c
        def power_law(x, a, b, c):
            return a * np.power(x, b) + c
        popt, pcov = curve_fit(power_law, x_data, y_data)
        fit_func = power_law
        params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
    elif law_type == 'logarithmic':
        # Logarithmic: y = a * log(x) + b
        def logarithmic(x, a, b):
            return a * np.log(x) + b
        popt, pcov = curve_fit(logarithmic, x_data, y_data)
        fit_func = logarithmic
        params = {'a': popt[0], 'b': popt[1]}
    else:
        raise ValueError(f'Unsupported law type: {law_type}')

    return {'params': params, 'fit_func': fit_func}

def plot_scaling_law(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_result: Dict[str, Any],
    x_label: str = 'Model Size (N)',
    y_label: str = 'Loss',
    title: str = 'Scaling Law Fit',
    save_path: str = None
):
    """
    Plots the scaling law fit against the data.

    Args:
        x_data: Independent variable data.
        y_data: Dependent variable data.
        fit_result: Result from fit_scaling_law function.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Plot title.
        save_path: Path to save the plot image (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, label='Data')

    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = fit_result['fit_func'](x_fit, *fit_result['params'].values())
    plt.plot(x_fit, y_fit, 'r-', label='Fit')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)

    if save_path:
        plt.savefig(save_path)
        print(f'Scaling law plot saved to {save_path}.')
    else:
        plt.show()
