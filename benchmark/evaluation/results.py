import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from scipy.stats import rankdata

def save_results(experiment_name: str, task_q_scores: dict[str, float], task_acc_scores: dict[str, float], output_dir: Path, config: dict = None):
    """Save raw results with timestamp and optional config snapshot."""
    
    # Cast output_dir to Path
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        
    result_path = output_dir / "results_summary.json"

    metadata = {
        "experiment": experiment_name,
        "overall_q_score": np.mean([v for v in task_q_scores.values()]),
        "overall_acc_score": np.mean([v for v in task_acc_scores.values()]),
        "task_q_scores": task_q_scores,
        "task_acc_scores": task_acc_scores,
    }
    if config:
        metadata["config"] = config

    with open(result_path, "w") as f:
        json.dump(metadata, f, indent=2)

def aggregate_results(output_dir: Path, phase: str) -> pd.DataFrame:
    """Aggregate all runs under a given phase into a DataFrame."""
    # Cast output_dir to Path
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
        
    phase_path = output_dir / phase
    if not phase_path.exists():
        raise FileNotFoundError(f"No results found under {phase_path}")

    rows = []
    for exp_dir in phase_path.iterdir():
        for json_file in exp_dir.glob("results_summary.json"):
            with open(json_file) as f:
                data = json.load(f)
                row = {
                    "experiment": data["experiment"],}

                row["overall_q_score"] = data.get("overall_q_score")
                row["overall_acc_score"] = data.get("overall_acc_score")

                # add per-task Q scores (used for ranking)
                for k, v in data.get("task_q_scores", {}).items():
                    row[f"q::{k}"] = v

                # store per-task acc scores (not used for ranking)
                for k, v in data.get("task_acc_scores", {}).items():
                    row[f"acc::{k}"] = v

                rows.append(row)

    return pd.DataFrame(rows)

def compute_leaderboard(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    """
    1. Exclude any runs that have NaN for any metric.
    2. Compute mean score across metrics.
    3. For each metric, rank experiments (highest→1).
    4. Compute weights = stddev per metric / sum(stddevs).
    5. Weighted_score = sum(rank * weight).
    6. aggregated_rank = rank(weighted_score).
    """
    df = df.copy()

    complete_mask = df[metric_columns].notna().all(axis=1)
    df = df.loc[complete_mask].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=["experiment", "mean_score", "weighted_score", "aggregated_rank"] + metric_columns
        )

    df["mean_score"] = df[metric_columns].mean(axis=1)

    # Per-metric ranks (highest better → negate)
    scores = df[metric_columns].to_numpy()
    per_task_ranks = rankdata(-scores, axis=0, method="min")

    # Metric weights from stddev
    if len(df) > 1:
        stds = np.std(scores, axis=0, ddof=1)
        weights = stds / stds.sum()
    else:
        weights = np.ones(len(metric_columns)) / len(metric_columns)

    df["weighted_score"] = (per_task_ranks * weights).sum(axis=1)
    df["aggregated_rank"] = rankdata(df["weighted_score"].values, method="min")

    # Sort best first
    cols = ["experiment", "mean_score", "weighted_score", "aggregated_rank"] + metric_columns
    return df.sort_values("weighted_score")[cols].reset_index(drop=True)

def save_leaderboard(df: pd.DataFrame, output_dir: Path, phase: str):
    """Save leaderboard summary CSV (including mean_score, weighted_score, aggregated_rank)."""
    # Cast output_dir to Path
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
    leaderboard_path = output_dir / "leaderboards" / phase
    leaderboard_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = leaderboard_path / f"leaderboard_{timestamp}.csv"
    df.to_csv(out_file, index=False)
    print(f"Leaderboard CSV saved to {out_file}")

def summarize_runs(output_dir: Path, phase: str):
    """Print and save leaderboard from all complete runs in a given phase."""
    df = aggregate_results(output_dir, phase)
    if df.empty:
        print("No results to summarize.")
        return

    q_metric_columns = [c for c in df.columns if c.startswith("q::")]
    leaderboard = compute_leaderboard(df, q_metric_columns)

    extra_cols = [c for c in df.columns if c.startswith("acc::")] + ["overall_q_score", "overall_acc_score"]
    extra_cols = [c for c in extra_cols if c in df.columns]

    leaderboard = leaderboard.merge(df[["experiment"] + extra_cols], on="experiment", how="left")

    if leaderboard.empty:
        print("No complete runs (all have missing metrics); nothing to rank.")
        return

    print("\n=== Leaderboard Summary ===")
    display_cols = (
            ["experiment", "mean_score", "weighted_score", "aggregated_rank"]
    )
    print(leaderboard[display_cols].to_string(index=False))

    save_leaderboard(leaderboard, output_dir, phase)
