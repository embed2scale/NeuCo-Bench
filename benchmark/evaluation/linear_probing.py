# Training and cross-validation pipeline for linear probes

import json
import copy
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit
from torchmetrics.classification import BinaryF1Score
from torchmetrics.regression import R2Score

from data.embeddings import EmbeddingDataset
from evaluation.metrics import classification_metrics, regression_metrics
from evaluation.visualisations import save_loss_curve, plot_confusion_matrix, plot_regression_scatter

logger = logging.getLogger(__name__)

@dataclass
class FoldResult:
    train_loss: list[float]
    val_loss: list[float]
    metric_history: list[float]
    best_metric: float
    best_model_state: dict

@dataclass
class TaskResult:
    q_statistic: float
    mean_score: float
    std_dev: float

# We employ a simple, linear model which can be used for regression and classification tasks.
# New models can be added for new downstream task in the future.
class LinearProbe(nn.Module):
    """Linear model for downstream tasks."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).view(-1)

class Trainer:
    """Train and evaluate a single fold."""

    def __init__(
                self,
                model: nn.Module,
                task_type: str,
                task_name: str,
                device: torch.device,
                learning_rate: float,
                fold_index: int,
                output_dir: Path,
                filename_prefix: str,
                enable_plots: bool,
            ) -> None:
        """Train and evaluate a single fold.
        
        Args:
            model: torch.Model to use in evaluation.
            task_type: Type of task, either "classification" or "regression".
            task_name: Name of task.
            device: Device, CPU or GPU, to run training and inference.
            learning_rate: Learning rate for Linear Probe training.
            fold_index: Index of split. Used in plotting.
            output_dir: Path to folder to store results in.
            filename_prefix: Prefix to add to files of visualizations.
            enable_plots: Flag controlling the creation of plots. Set to True to plot results.
        """
        self.model = model.to(device)
        self.task_type = task_type
        self.task_name = task_name
        self.device = device
        self.fold_index = fold_index
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.enable_plots = enable_plots
 
        if task_type in ("classification", "cls"):
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.metric_fn = BinaryF1Score().to(device)
        elif task_type in ("regression", "regr"):
            self.loss_fn = nn.MSELoss()
            self.metric_fn = R2Score().to(device)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _step(
            self,
            batch: tuple[torch.Tensor, torch.Tensor],
            training: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = (tensor.to(self.device) for tensor in batch)
        with torch.set_grad_enabled(training):
            preds = self.model(inputs)
            loss = self.loss_fn(preds, targets.float())
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return loss.detach(), preds.detach()

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
        ) -> FoldResult:
        """Fit model to data and evaluate.

        Args:
            train_loader: Data loader for training data.
            val_loader: Data loader for validation data.
            epochs: Number of epochs to train network.
        Returns:
            Instance of FoldResult containing the evaluation results.
        """
        train_losses, val_losses, metrics = [], [], []
        best_metric = float("-inf")
        best_state = None
        best_train_preds, best_train_targets = None, None
        best_val_preds, best_val_targets = None, None

        for _ in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_losses, train_preds, train_targets = [], [], []
            for batch in train_loader:
                loss, preds = self._step(batch, training=True)
                epoch_train_losses.append(loss)
                train_preds.append(preds)
                train_targets.append(batch[1])
            train_losses.append(torch.stack(epoch_train_losses).mean().item())

            # Validation phase
            self.model.eval()
            epoch_val_losses, val_preds, val_targets = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    loss, preds = self._step(batch, training=False)
                    epoch_val_losses.append(loss)
                    val_preds.append(preds)
                    val_targets.append(batch[1])
            val_losses.append(torch.stack(epoch_val_losses).mean().item())
            
            # Compute metric on validation data
            metric_value = self.metric_fn(
                torch.cat(val_preds), torch.cat(val_targets)
            ).item()
            metrics.append(metric_value)

            # Update best model
            if metric_value > best_metric:
                best_metric = metric_value
                best_state = copy.deepcopy(self.model.state_dict())
                best_train_preds, best_train_targets = train_preds, train_targets
                best_val_preds, best_val_targets = val_preds, val_targets

        # Generate plots if enabled
        if self.enable_plots:
            if self.task_type in ("classification", "cls"):
                cm = classification_metrics(
                    torch.cat(best_train_preds), torch.cat(best_train_targets)
                )["confusion_matrix"]
                plot_confusion_matrix(
                    np.array(cm), self.task_name, self.fold_index,
                    self.output_dir, self.filename_prefix
                )
            else:
                plot_regression_scatter(
                    torch.cat(best_train_targets), torch.cat(best_train_preds),
                    torch.cat(best_val_targets), torch.cat(best_val_preds),
                    self.task_name, self.fold_index,
                    self.output_dir, self.filename_prefix,
                )

        return FoldResult(train_losses, val_losses, metrics, best_metric, best_state)

def cross_validate(
        df: pd.DataFrame,
        task_type: str,
        task_name: str,
        device: torch.device,
        batch_size: int,
        n_splits: int,
        epochs: int,
        embedding_dim: int,
        learning_rate: float,
        output_dir: Path,
        filename_prefix: str,
        enable_plots: bool,
        random_seed: int = 42,
        ) ->  TaskResult:
    """Perform k-fold cross-validation with a linear probe.

    Args:
        df: Dataframe to evaluate.
        task_type: Type of task, either "classification" or "regression".
        task_name: Name of task.
        device: Device, CPU or GPU, to run training and inference.
        batch_size: Batch size in Linear Probe training.
        n_splits: Number of repetitions of Linear Probe evaluation. Use to gather statistics.
        epochs: Number of epochs to use in Linear Probe training.
        embedding_dim: Size of embeddings.
        learning_rate: Learning rate for Linear Probe training.
        output_dir: Path to folder to 
    Returns:
        A TaskResult instance containing the evaluation results.
    """
    logger.info("Cross-validation start: %s", task_name)

    dataset = EmbeddingDataset(df)
    splitter = ShuffleSplit(
        n_splits=n_splits, test_size=0.1, random_state=random_seed
    )
    fold_results: list[FoldResult] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(dataset)):
        logger.info("Fold %d/%d", fold_index + 1, n_splits)

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(test_idx),
            pin_memory=True,
        )

        model = LinearProbe(embedding_dim)
        trainer = Trainer(
            model=model,
            task_type=task_type,
            task_name=task_name,
            device=device,
            learning_rate=learning_rate,
            fold_index=fold_index,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            enable_plots=enable_plots,
        )
        fold_results.append(trainer.fit(train_loader, val_loader, epochs))
    
    # Aggregate and save results
    train_losses = [fr.train_loss for fr in fold_results]
    val_losses = [fr.val_loss for fr in fold_results]
    best_metrics = np.array([fr.best_metric for fr in fold_results], dtype=np.float64)

    save_loss_curve(train_losses, output_dir, task_name, loss_type="train")
    save_loss_curve(val_losses, output_dir, task_name, loss_type="validation")

    # Compute summary Q-statistic
    mean_score = np.nanmean(best_metrics)
    std_dev = np.nanstd(best_metrics)
    q_stat = mean_score / (0.02 + std_dev) * 2

    result_metrics = {
        "q_stat": q_stat,
        "mean_score": mean_score,
        "std_dev": std_dev,
    }

    (output_dir / task_name / f"{task_name}_result.json").write_text(
        json.dumps(result_metrics, indent=2)
    )
    
    return TaskResult(q_stat, mean_score, std_dev)
