import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import torch

from data.embeddings import load_submission
from data.labels import get_annotations
from evaluation.linear_probing import cross_validate
from evaluation.results import save_results, summarize_runs
from evaluation.utils import fix_all_seeds

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
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Determine device
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.get("device", "auto") == "auto"
        else torch.device(config["device"])
    )

    # Load data
    annotation_df = get_annotations(annotation_path)
    submission_df = load_submission(
        file_path=submission_file,
        valid_ids=set(annotation_df['id']),
        expected_dim=config['embedding_dim'],
        exclude_file=exclude_file,
        standardize=config['standardize_embeddings']
    )

    merged_df = annotation_df.merge(submission_df, on="id").dropna(subset=["embedding"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{method_name}_{timestamp}"
    run_dir = output_dir / phase / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    task_results = {}
    task_filter = config.get("task_filter")

    for task_name, group in merged_df.groupby("task_name"):
        if task_filter is not False and task_name not in task_filter:
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
            device=device,
            batch_size=config["batch_size"],
            n_splits=config["k_folds"],
            epochs=config["epochs"],
            embedding_dim=config["embedding_dim"],
            learning_rate=config["learning_rate"],
            output_dir=run_dir,
            filename_prefix=submission_file.stem,
            enable_plots=config.get("enable_plots", True),
        )

        task_results[task_name] = result.q_statistic

    save_results(experiment_name=experiment_name, task_results=task_results, output_dir=run_dir, config=config)
    logger.info("Finished evaluation.")
