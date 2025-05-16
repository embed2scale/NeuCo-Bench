import argparse
import logging
from pathlib import Path
import yaml

from evaluation.evaluation import evaluate
from evaluation.results import summarize_runs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_file", type=Path, required=True, description='File containing compressed embeddings to evaluate.')
    parser.add_argument("--exclude_file", type=Path, default=None, required=False, description='File containing your compressed embeddings.')
    parser.add_argument("--annotation_path", type=Path, required=True, description='Folder containing csv label files per downstream task.')
    parser.add_argument("--config", type=Path, default="config.yaml", description='YAML file with cross-validation settings, and logging options. See provided sample config.')
    parser.add_argument("--method_name", type=str, required=True, description='Identifier for your compression method—used to tag outputs and leaderboards.')
    parser.add_argument("--output_dir", type=Path, default=Path("results/"), description='Directory to save per-task reports, plots, and aggregated results.')
    parser.add_argument("--phase", type=str, default="all", description='A label (e.g., “dev”, “eval”) defining a particular benchmark setup. Results for each phase are stored in a separate subfolder under output_dir.')
    return parser.parse_args()

def main():
    args = parse_args()
    with args.config.open() as f:
        config = yaml.safe_load(f)

    evaluate(
            submission_file=args.submission_file,
            exclude_file=args.exclude_file,
            annotation_path=args.annotation_path,
            method_name=args.method_name,
            output_dir=args.output_dir,
            phase=args.phase,
            config=config,
        )

    if config["update_leaderboard"]:
        summarize_runs(output_dir=args.output_dir, phase=args.phase)

if __name__ == "__main__":
    main()
