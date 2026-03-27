
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics.classification import BinaryF1Score
from torchmetrics.regression import R2Score

from data.embeddings import EmbeddingDataset
from evaluation.metrics import classification_metrics
from evaluation.utils import FoldResult
from evaluation.visualisations import plot_confusion_matrix, plot_regression_scatter

class LinearProbe(nn.Module):
    """Linear model for downstream tasks."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).view(-1)


class LinearTrainer:
    """Train and evaluate a single fold."""

    def __init__(
                self,
                task_type: str,
                task_name: str,
                embedding_dim: int,
                device: torch.device,
                fold_index: int,
                output_dir: Path,
                filename_prefix: str,
                enable_plots: bool,
                learning_rate: float = 0.001,
                epochs: int = 10,
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
            epochs: Default number of epochs to train network.
        """

        self.init_params = {'input_dim': embedding_dim}
        self.model = LinearProbe(**self.init_params)
        self.task_type = task_type
        self.task_name = task_name
        self.device = device
        self.fold_index = fold_index
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.enable_plots = enable_plots
        self.epochs = epochs
 
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
    
    def fit_only(self, 
            data_loader: DataLoader,
            epochs: int = None,
            refit: bool = False,
            ):
        if refit:
            self.model = LinearProbe(**self.init_params)

        for _ in range(epochs):
            # Training phase
            self.model.train()
            for batch in data_loader:
                _, _ = self._step(batch, training=True)

    def fit_evaluate(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            epochs: int = None,
        ) -> FoldResult:
        """Fit model to data and evaluate.

        Args:
            train_loader: Data loader for training data.
            val_loader: Data loader for validation data.
            epochs: Number of epochs to train network.
        Returns:
            Instance of FoldResult containing the evaluation results.
        """

        if epochs is None:
            epochs = self.epochs

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

            if val_loader is not None:
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
                    np.array(cm), self.task_name, 
                    self.output_dir, self.filename_prefix,
                    fold_idx=self.fold_index,
                )
            else:
                plot_regression_scatter(
                    torch.cat(best_train_targets), torch.cat(best_train_preds),
                    self.task_name, 
                    self.output_dir, self.filename_prefix,
                    fold_idx=self.fold_index,
                    y_val_true=torch.cat(best_val_targets), 
                    y_val_pred=torch.cat(best_val_preds),
                )

        self.best_metric = best_metric
        return FoldResult(train_losses, val_losses, metrics, best_metric, best_state, self.model)

    def save_model(self, path: Path) -> None:
        """Save the best-fitted model and write a Python script to load it.

        Args:
            path: Directory path where model files will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights and metadata
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.model.linear.in_features,
            'output_dim': self.model.linear.out_features,
            'task_type': self.task_type,
        }, path / 'model.pth')

        # Write load script (embeds LinearProbe source for portability)
        load_script = path / 'load_model.py'
        if not load_script.exists():
            import inspect
            linear_source = inspect.getsource(LinearProbe)
            load_script.write_text(f'''# -*- coding: utf-8 -*-
"""Auto-generated, self-contained script to load the saved linear probe model."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

{linear_source}


def load_model(checkpoint_path: Path, device: torch.device = None) -> tuple:
    """Load a saved linear probe model.

    Args:
        checkpoint_path: Path to the directory containing model.pth and load_model.py.
        device: Device to load the model onto. Defaults to cuda if available.

    Returns:
        Tuple of (model, task_type).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path / "model.pth", map_location=device)

    model = LinearProbe(checkpoint["input_dim"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["task_type"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a saved linear probe model.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to the checkpoint directory.")
    args = parser.parse_args()
    model, task_type = load_model(args.checkpoint.resolve())
    print(f"Model loaded. Task type: {{task_type}}")
    print(f"Model device: {{next(model.parameters()).device}}")
''')
    

class LinearFoldRunner():
    """Trains linear probe for multiple folds, each with its own LInearTrainer."""
    
    def __init__(self, 
                 df: pd.DataFrame,
               task_type,
               task_name,
               device,
               output_dir,
               filename_prefix,
               enable_plots,
               embedding_dim,
               probe_params,
               splitter,
               store_models,
               ):
        
        self.task_type = task_type
        self.task_name = task_name
        self.device = device
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.enable_plots = enable_plots
        self.embedding_dim = embedding_dim
        self.probe_params = probe_params
        self.splitter = splitter
        self.store_models = store_models
        
        self.dataset = EmbeddingDataset(df)
        self.trainer = None
        
        self.batch_size = self.probe_params.pop('batch_size', 64)
        self.epochs = self.probe_params.get('epochs', 10)

        self.hyperparams = {'epochs':self.epochs,
                            'learning_rate': self.probe_params.get('learning_rate', 0.001),
                            'batch_size': self.batch_size,
                            }

    def train(self, x=None, y=None):
        """
        Run training over folds.

        Args:
            x (None): Ignored. Exists for API compatibility.
            y (None): Ignored. Exists for API compatibility.
        
        Returns:
            List[FoldResults]: List of results from training folds.
            LinearTrainer: Trainer from final fold.
        """

        fold_results: list[FoldResult] = []

        # Train and evaluate over fold
        for fold_index, (train_idx, test_idx) in tqdm(enumerate(self.splitter.split(self.dataset)), 
                                                      total=self.splitter.get_n_splits()):
            
            train_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(train_idx),
                pin_memory=True,
            )
            val_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=SubsetRandomSampler(test_idx),
                pin_memory=True,
            )

            trainer = LinearTrainer(
                task_type=self.task_type,
                task_name=self.task_name,
                device=self.device,
                fold_index=fold_index,
                output_dir=self.output_dir,
                filename_prefix=self.filename_prefix,
                enable_plots=self.enable_plots,
                embedding_dim=self.embedding_dim,
                **self.probe_params,
            )
            fold_results.append(trainer.fit_evaluate(train_loader, val_loader=val_loader, epochs=self.epochs))
            self.trainer = trainer

        # Retrain on all data
        if self.store_models:
            data_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                pin_memory=True,
            )
            trainer.fit_only(data_loader, epochs=self.epochs, refit=True)
            self.trainer = trainer
            
        return fold_results
    
    def save_model(self, path: Path):
        self.trainer.save_model(path)
    
    @property
    def model(self):
        return self.trainer.model