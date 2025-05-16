from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

def save_plot(
    fig: plt.Figure,
    folder: Path,
    filename: str,
    dpi: int = 300,
) -> None:
    # Ensure target directory exists
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / filename
    # Save and close 
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def save_loss_curve(
    loss_curves: Sequence[Sequence[float]],
    output_folder: Path,
    task_name: str,
    loss_type: str = "train",
) -> None:
    """
    Plot and save one or multiple loss curves (training and/or validation).

    Args:
        loss_curves: An iterable of loss sequences, e.g., [train_loss, val_loss].
        output_folder: Directory where the plot will be saved.
        task_name: Identifier for the current task.
        loss_type: Descriptor for the loss type (e.g., 'train', 'val', or 'combined').
    """
    # Build target directory for loss plots
    target_folder = output_folder / task_name / "loss_curves"
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each loss curve
    for idx, curve in enumerate(loss_curves, start=1):
        ax.plot(curve, label=f"Curve {idx}") 

    ax.set_title(f"Loss Curves - {task_name} ({loss_type})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)

    # Save figure
    filename = f"{task_name}_{loss_type}_loss_curves.png"
    save_plot(fig, target_folder, filename)

def plot_regression_scatter(
    y_train_true: Sequence[float],
    y_train_pred: Sequence[float],
    y_val_true: Sequence[float],
    y_val_pred: Sequence[float],
    task_name: str,
    fold_idx: int,
    output_folder: Path,
    base_name: str,
) -> None:
    """
    Generate and save a scatter plot comparing true vs. predicted values for regression.

    Args:
        y_train_true: True target values for the training set.
        y_train_pred: Predicted values for the training set.
        y_val_true: True target values for the validation set.
        y_val_pred: Predicted values for the validation set.
        task_name: Identifier for the current task.
        fold_idx: Index of the cross-validation fold (0-based).
        output_folder: Root folder for saving results.
        base_name: Base filename prefix.
    """
    # Build target directory for scatter plots
    target_folder = output_folder / task_name / "regression_scatter"
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot training and validation points
    ax.scatter(y_train_true, y_train_pred, alpha=0.3, label="Train")
    ax.scatter(y_val_true, y_val_pred, alpha=0.6, label="Validation")

    # Draw diagonal identity line
    bounds = [min(min(y_train_true), min(y_val_true)), max(max(y_train_true), max(y_val_true))]
    ax.plot(bounds, bounds, linestyle='--', linewidth=2)

    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    ax.set_title(f"Regression Scatter: {task_name}, Fold {fold_idx + 1}")

    # Save figure
    filename = f"{base_name}_{task_name}_scatter_fold{fold_idx + 1}.png"
    save_plot(fig, target_folder, filename)


def plot_confusion_matrix(
    cm_array: np.ndarray,
    task_name: str,
    fold_idx: int,
    output_folder: Path,
    base_name: str,
) -> None:
    """
    Generate and save a confusion matrix plot for a classification task.

    Args:
        cm_array: Square confusion matrix as a NumPy array.
        task_name: Identifier for the current task.
        fold_idx: Index of the cross-validation fold (0-based).
        output_folder: Root folder for saving results.
        base_name: Base filename prefix.
    """
    # Build target directory
    target_folder = output_folder / task_name / "confusion_matrices"
    fig, ax = plt.subplots(figsize=(6, 5))

    # Display confusion matrix
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix: {task_name}, Fold {fold_idx + 1}")
    fig.colorbar(im, ax=ax)

    # Set tick labels
    labels = np.arange(cm_array.shape[0])
    ax.set_xticks(labels)
    ax.set_yticks(labels)

    # Annotate each cell 
    thresh = cm_array.max() / 2.0
    for i in labels:
        for j in labels:
            color = 'white' if cm_array[i, j] > thresh else 'black'
            ax.text(j, i, f"{cm_array[i, j]}", ha='center', va='center', color=color)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    
    # Save figure
    filename = f"{base_name}_{task_name}_cm_fold{fold_idx + 1}.png"
    save_plot(fig, target_folder, filename)



