# NeuCo-Bench

**Licence**: Apache-2.0

*Originally developed to evaluate challenge submissions for the 2025 EARTHVISION Challenge at CVPR ([competition details](https://www.grss-ieee.org/events/earthvision-2025/?tab=challenge)), NeuCo-Bench is now released for local benchmarking and evaluation.*

NeuCo-Bench is a **benchmarking framework** designed to evaluate how effectively compressed embeddings preserve information for downstream tasks.

In domains like Earth Observation (EO), pipelines typically handle large volumes of image data used primarily for analytical tasks. Traditional compression techniques focus on pixel-level reconstruction, while Foundation Model (FM) research does not explicitly consider embedding size. NeuCo-Bench addresses this gap by enforcing strict size constraints and evaluating embeddings directly on real-world EO tasks.

NeuCo-Bench provides an initial set of EO tasks and invites community contributions of additional tasks and datasets from EO and other domains.

<p align="center">
  <img src="assets/NeuCoBench.png" alt="Framework overview" width="100%" />
</p>


## Key Features

- **Model-agnostic**: Supports evaluation of any fixed-size embedding (e.g. 1024‑dim feature vectors), which enables comparison among compression and representation learning methods.
- **Task-Driven Evaluation**: Utilizes linear probes across diverse EO tasks, including land-cover proportion estimation, cloud detection, and biomass estimation. 
- **Metrics**: Incorporates signal-to-noise scores and dynamic rank aggregation to compare methods.

---

## Quickstart

```bash
# start from fresh environment (skip if not needed)
micromamba create -n neuco-bench -c conda-forge python=3.12
micromamba activate neuco-bench

# clone NeuCo-Bench and install requirements
git clone https://github.com/embed2scale/NeuCo-Bench.git
cd NeuCo-Bench/benchmark
pip install -r ../requirements.txt

# run standalone NeuCo-Bench evaluation script
python main.py \
  --annotation_path path/to/annotation_folder \
  --submission_file path/to/submission_file.csv \
  --output_dir path/to/results \
  --config path/to/config.yaml \
  --method_name your-method-name \
  --phase phase-name
```

- `--annotation_path` Directory containing CSV label files for each task.  
- `--submission_file` CSV file with your embeddings.  
- `--output_dir` Destination for per-task reports, plots, and aggregated benchmark results.  
- `--config` YAML file specifying cross-validation settings and logging options (see provided sample).  
- `--method_name` Identifier for your method used in filenames and leaderboard entries.  
- `--phase` Groups evaluation runs under a specified phase name for ranking, creating a subfolder within `output_dir`. 

To disable GPU utilization, run `CUDA_VISIBLE_DEVICES=''` before execution.

## Overview

NeuCo-Bench emphasizes task-oriented semantic evaluation rather than pixel-level reconstruction, measuring how effectively compressed embeddings retain information relevant to EO tasks.

To evaluate embeddings:
1. Download the [SSL4EO-S12-downstream dataset](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) from Hugging Face (see [Data](#data)).  
2. Encode images into fixed-size embeddings, save as CSV (see [Creating Embeddings](#creating-embeddings)).  
3. Run NeuCo-Bench locally to evaluate and aggregate scores, generating a leaderboard (see [Evaluation and Ranking](#evaluation-and-ranking)).

---

## Data

The **SSL4EO-S12-downstream** dataset includes:

- `data/`  
  Subfolders for modalities (`s1/`, `s2l1c/`, `s2l2a/`) with subsets of 1000 `zarr.zip` files each.
- `labels/`  
  Annotation files for each downstream task.

Both `data/` and `labels/` are required. See `examples/data` for a TorchDataset loader; if you experience data-loading errors, verify that `zarr==2.18.0` is used.

Data format aligns with [SSL4EOS12 v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1), recommended as a pretraining dataset.

---

## Creating Embeddings

Generate embeddings and save them as CSV files. Example scripts in `examples/` illustrate the required format and provide two baseline methods: Averaging Baseline (Bilinear interpolation and averaging of the modalities) and downsampled embeddings from a pretrained FM (DINO ViT pretrained on SSL4EO).

To ensure consistent benchmarking, all methods should use the same embedding dimension. We set the embedding size to 1024 (dimensions) during the 2025 CVPR EARTHVISION data challenge.
As reference, we provide a selection of CSV files from the 2025 CVPR EARTHVISION data challenge in the repo's top-level `data/` directory. More details in `data/README.md`.
In general, the https://github.com/embed2scale/NeuCo-Bench/tree/main/data folder is tracked by Git LFS to keep initial clones of this repo slim. If you like to download the approx. 500 MB of embeddings, utilize:
```Bash
git lfs install
git pull
```

---

## Evaluation and Ranking

Run the benchmark on your embeddings with:

```bash
python main.py \
  --annotation_path path/to/annotation_folder \
  --submission_file path/to/submission_file.csv \
  --output_dir path/to/results \
  --config path/to/config.yaml \
  --method_name "your-method-name" \
  --phase "phase-name"
```

### Configuration

A sample config file (`benchmark/config.yaml`) specifies:

- `batch_size`, `epochs`, `learning_rate`, `k_folds`: Cross-validation settings. 
- `standardize_embeddings`: Standardize embeddings using global mean and std (recommended).  
- `normalize_labels`: Normalize target labels to [0,1] (recommended).
- `enable_plots`: Generate per-fold plots (e.g., parity plots for regression).  
- `update_leaderboard`: Aggregate and update leaderboard after evaluation.  
- `task_filter`: Tasks to evaluate (default: all tasks available in `annotation_path`). 

### Results

Results saved under `output_dir/<phase-name>/` include:

- Task-specific metrics and loss curves
- `results_summary.json` with per-task signal-to-noise scores and overall scores

### Aggregation

Aggregate scores for leaderboard by setting `update_leaderboard` to `True` during last evaluation or manually run:

```bash
from evaluation.results import summarize_runs
summarize_runs(output_dir=output_dir, phase=phase)
```

---

## Future Work & Contributing

All downstream tasks and labels are published on Hugging Face. We are planning to extend the framework to further tasks (eg. spatial and temporal downstream tasks).

We invite the community to collaborate and appreciate contributions, including but not limited to the following:
- Benchmark and contribute new compression techniques
- Incorporate additional downstream task and metrics
- Extension to further input modalities

Check out [CONTRIBUTING.md](.github/CONTRIBUTING.md).
