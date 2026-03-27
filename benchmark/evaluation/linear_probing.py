# Training and cross-validation pipeline for linear probes

import json
import os
import logging
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

from evaluation.probes.linear import LinearFoldRunner
from evaluation.probes.svm import SVMHPOptimizer
from evaluation.visualisations import save_loss_curve
from evaluation.utils import TaskResult

logger = logging.getLogger(__name__)


def cross_validate(
        df: pd.DataFrame,
        task_type: str,
        task_name: str,
        device: torch.device,
        n_splits: int,
        embedding_dim: int,
        output_dir: Path,
        filename_prefix: str,
        enable_plots: bool,
        random_seed: int = 42,
        output_fold_results: Optional[bool] = False,
        store_models: Optional[bool] = False,
        probe_type: Optional[str] = 'linear',
        probe_params: Optional[dict] = None,
        ) ->  TaskResult:
    """Perform shuffle split validation with a linear probe. 
    Returns the trained model from the last split in the return object.

    Args:
        df (pandas.DataFrame): Dataframe to evaluate.
        task_type (str): Type of task, either "classification" or "regression".
        task_name (str): Name of task.
        device (torch.device): Device, CPU or GPU, to run training and inference.
        n_splits (int): Number of repetitions of Linear Probe evaluation. Use to gather statistics.
        embedding_dim (int): Size of embeddings.
        output_dir (str): Path to folder to store results in.
        enable_plots (bool): Toggle storing of plots. Set to True to store plots.
        random_seed (int): Integer seed for random number generator.
        output_fold_results (bool, optional): Toggle storing performance metric per fold in addition to summary statistics. Default is False, in which case performance per fold is not stored.
        store_models (bool optional): Toggle storing model under <output_dir>/models/<task_name>. Default is False. The model from the final fold is stored.
        probe_type (str, optional): String with the probe type. Currently implemented are 'linear' and 'svm', the former using a single linear layer and the second an SVM.
        probe_params (dict, optional): Dictionary with parameters for probe.
    Returns:
        TaskResult: A TaskResult instance containing the evaluation results.
    """

    logger.info("Cross-validation start: %s", task_name)

    assert probe_type in ['linear', 'svm'], f'Probe type must be "svm" or "linear", but was {probe_type}.'
    assert output_dir is not None, f'output_dir cannot be empty.'

    if probe_params is None:
        probe_params = {}

    splitter = ShuffleSplit(
        n_splits=n_splits, test_size=0.1, random_state=random_seed
    )
    loss_curve_x_label = None
    if probe_type == 'linear':
        loss_curve_x_label = "Epoch"

        trainer = LinearFoldRunner(df=df,
                                   task_type=task_type,
                                   task_name=task_name,
                                   device=device,
                                   output_dir=output_dir,
                                   filename_prefix=filename_prefix,
                                   enable_plots=enable_plots,
                                   embedding_dim=embedding_dim,
                                   probe_params=probe_params,
                                   splitter=splitter,
                                   store_models=store_models,
                                   )
        
        fold_results = trainer.train()
        model = trainer.model
        best_params = trainer.hyperparams

    elif probe_type == 'svm':
        loss_curve_x_label = "Hyperparameter setting"
        
        # Set random seed for optimizer if not already defined
        opt_params = probe_params.get('opt_params', None)
        if opt_params is not None:
            if 'random_state' not in opt_params:
                opt_params['random_state'] = random_seed
                probe_params['opt_params'] = opt_params

        trainer = SVMHPOptimizer(
            df=df,
            task_type=task_type,
            task_name=task_name,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
            enable_plots=enable_plots,
            splitter=splitter,
            **probe_params,
        )
        fold_results = trainer.train()
        model = trainer.model
        best_params = trainer.best_model_state

    # Aggregate and save results
    train_losses = [fr.train_loss for fr in fold_results]
    val_losses = [fr.val_loss for fr in fold_results]
    best_metrics = np.array([fr.best_metric for fr in fold_results], dtype=np.float64)

    save_loss_curve(train_losses, output_dir, task_name, loss_type="train", xlabel=loss_curve_x_label,)
    save_loss_curve(val_losses, output_dir, task_name, loss_type="validation", xlabel=loss_curve_x_label,)

    # Compute summary Q-statistic
    mean_score = np.nanmean(best_metrics)
    std_dev = np.nanstd(best_metrics)
    q_stat = mean_score / (0.02 + std_dev) * 2

    result_metrics = {
        "q_stat": q_stat,
        "mean_score": mean_score,
        "std_dev": std_dev,
        "hyperparamters": best_params,
    }
    if output_fold_results:
        result_metrics["q_t"] = best_metrics.tolist()

    (output_dir / task_name / f"{task_name}_result.json").write_text(
        json.dumps(result_metrics, indent=2)
    )

    # Store models
    if store_models:
        model_save_path = output_dir / task_name / 'model'
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)

    return TaskResult(task_name, q_stat, mean_score, std_dev, model)
