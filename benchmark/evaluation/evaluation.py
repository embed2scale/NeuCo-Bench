import json
import logging
import os
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import ShuffleSplit

from data.embeddings import load_submission
from data.labels import get_annotations
from evaluation.probes.linear import LinearFoldRunner
from evaluation.probes.svm import SVMHPOptimizer
from evaluation.results import save_results
from evaluation.utils import fix_all_seeds, TaskResult
from evaluation.visualisations import save_loss_curve


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmarking")


def evaluate(submission_file: Path, 
             annotation_path: Path, 
             method_name: str, 
             output_dir: Path, 
             phase: str, 
             config: dict, 
             exclude_file: str = None) -> None:
    """Evaluate an set of embeddings on one or multiple tasks. 
    Results are written to output_dir folder.

    Args:
        submission_file: Path to the submission file containing embeddings.
        annotation_path: Path to folder containing label files
        method_name: Name of experiment. Used to distinguish methods compared in the same experiment.
        output_dir: Path to folder to which results are written.
        phase: Name of phase. Used to distinguish experiments.
        config: Dictionary containing evaluation configurations.
        exclude_file: Path to file containing embedding IDs to exclude. If not provided, exclude no embeddings.
    """
    fix_all_seeds(seed=42)

    # Load data
    annotation_df = get_annotations(annotation_path)
    submission_df, embedding_dim = load_submission(
        file_path=submission_file,
        valid_ids=set(annotation_df['id']),
        expected_dim=config.get("embedding_dim", None),
        exclude_file=exclude_file,
        standardize=config.get('standardize_embeddings', True)
    )

    merged_df = annotation_df.merge(submission_df, on="id").dropna(subset=["embedding"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_{timestamp}"
    run_dir = output_dir / phase / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    task_q_scores = {}
    task_acc_scores = {}
    task_filter = config.get("task_filter", None)

    for task_name, group in merged_df.groupby("task_name"):
        if task_filter is not None and task_name not in task_filter:
            logger.info("Skipping task %s due to filter", task_name)
            continue

        task_type = group["task_type"].iloc[0]

        if config.get("normalize_labels", True):
            group = group.copy()
            group["label"] = (group["label"] - group["label"].min()) / (group["label"].max() - group["label"].min())

        logger.info("Evaluating %s (%s)", task_name, task_type)

        result = cross_validate(
            df=group,
            task_type=task_type,
            task_name=task_name,
            probe_type=config["probe_type"],
            probe_params=config["probe_params"],
            n_splits=config["k_folds"],
            embedding_dim=embedding_dim,
            output_dir=run_dir,
            filename_prefix=submission_file.stem,
            enable_plots=config.get("enable_plots", True),
            output_fold_results=config.get("output_fold_results", False),
            store_probes=config.get('store_probes', False),
        )

        task_q_scores[task_name] = result.q_statistic
        task_acc_scores[task_name] = result.mean_score

    save_results(experiment_name=experiment_name,
                 task_q_scores=task_q_scores,
                 task_acc_scores=task_acc_scores,
                 output_dir=run_dir, 
                 config=config,
                 )

    logger.info("Finished evaluation.")


def cross_validate(
        df: pd.DataFrame,
        task_type: str,
        task_name: str,
        probe_type: str,
        probe_params: dict,
        n_splits: int,
        embedding_dim: int,
        output_dir: Path,
        filename_prefix: str,
        enable_plots: bool,
        random_seed: int = 42,
        output_fold_results: Optional[bool] = False,
        store_probes: Optional[bool] = False,
        ) ->  TaskResult:
    """Perform shuffle split validation with a linear probe. 
    Returns the trained model from the last split in the return object.

    Args:
        df (pandas.DataFrame): Dataframe to evaluate.
        task_type (str): Type of task, either "classification" or "regression".
        task_name (str): Name of task.
        probe_type (str): String with the probe type. Currently implemented are 'linear' and 'svm', the former using a single linear layer and the second an SVM.
        probe_params (dict): Dictionary with parameters for probe.
        device (torch.device): Device, CPU or GPU, to run training and inference.
        n_splits (int): Number of repetitions of Linear Probe evaluation. Use to gather statistics.
        embedding_dim (int): Size of embeddings.
        output_dir (str): Path to folder to store results in.
        enable_plots (bool): Toggle storing of plots. Set to True to store plots.
        random_seed (int): Integer seed for random number generator.
        output_fold_results (bool, optional): Toggle storing performance metric per fold in addition to summary statistics. Default is False, in which case performance per fold is not stored.
        store_probes (bool, optional): Toggle storing model under <output_dir>/models/<task_name>. Default is False. The model from the final fold is stored.
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

        # Determine device
        device = probe_params.pop("device", "auto")
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        
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
                                   store_probes=store_probes,
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
        "hyperparamaters": best_params,
    }
    if output_fold_results:
        result_metrics["q_t"] = best_metrics.tolist()

    (output_dir / task_name / f"{task_name}_result.json").write_text(
        json.dumps(result_metrics, indent=2)
    )

    # Store models
    if store_probes:
        model_save_path = output_dir / task_name / 'probe'
        os.makedirs(model_save_path, exist_ok=True)
        trainer.save_model(model_save_path)

    return TaskResult(task_name, q_stat, mean_score, std_dev, model)
