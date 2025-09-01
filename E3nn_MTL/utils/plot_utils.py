import matplotlib.pyplot as plt
import torch
import os
from pathlib import Path

def plot_training_curves(train_loss, valid_loss, log_scale=False, title=None, save_path=None):
    """
    Plot training and validation loss curves with customizable options and save capability.
    
    Args:
        train_loss (torch.Tensor or numpy.ndarray): Training loss values
        valid_loss (torch.Tensor or numpy.ndarray): Validation loss values
        log_scale (bool): Whether to use logarithmic scale for y-axis (default: False)
        title (str): Custom title for the plot. If None, uses default title (default: None)
        save_path (str): Full path including filename to save the plot (e.g., '/home/folder/myname.png'). 
                        If None, plot is not saved (default: None)
    """
    # Convert tensors to numpy arrays if they're pytorch tensors
    if hasattr(train_loss, 'numpy'):
        train_loss = train_loss.numpy()
    if hasattr(valid_loss, 'numpy'):
        valid_loss = valid_loss.numpy()
    
    fig = plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    
    # Set title
    default_title = 'Training and Validation Loss Curves'
    if log_scale:
        default_title += ' (Log Scale)'
    
    plt.title(title if title is not None else default_title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Set logarithmic scale if requested
    if log_scale:
        plt.yscale('log')
    
    # Show grid (more visible lines for log scale)
    plt.grid(True, linestyle='--', alpha=0.5, which='both' if log_scale else 'major')
    
    # Save the plot if save_path is provided
    if save_path is not None:
        # Extract directory from path and create if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory:  # Only try to create if directory path exists (not just filename)
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save with tight layout to prevent label cutoff
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    plt.close(fig)


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def plot_predictions_vs_real(real_values, predictions, title=None, save_path=None):
    """
    Plot predictions against real values with a perfect prediction line.
    
    Args:
        real_values (torch.Tensor or numpy.ndarray): Ground truth values
        predictions (torch.Tensor or numpy.ndarray): Model predictions
        title (str): Custom title for the plot (default: 'Predictions vs Real Values')
        save_path (str): Full path to save the plot (e.g., '/path/to/plot.png'). If None, plot is not saved.
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(real_values, 'numpy'):
        real_values = real_values.numpy()
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    
    # Flatten arrays in case they're not 1D
    real_values = real_values.flatten()
    predictions = predictions.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot perfect prediction line (y = x)
    min_val = min(real_values.min(), predictions.min())
    max_val = max(real_values.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', label='Perfect Prediction', alpha=0.7)
    
    # Scatter plot of actual predictions
    ax.scatter(real_values, predictions, 
              alpha=0.5, label='Model Predictions', color='blue')
    
    # Set plot details
    ax.set_xlabel('Real Values', fontsize=12)
    ax.set_ylabel('Predictions', fontsize=12)
    ax.set_title(title if title else 'Predictions vs Real Values', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Save if path is provided
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory:  # Only create if directory path exists
            Path(directory).mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    plt.close(fig)