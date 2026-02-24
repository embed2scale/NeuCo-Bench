# Evaluation and Results

This page describes how NeuCo-Bench evaluates embeddings, how to configure the pipeline, how results are stored, and how to aggregate runs into a leaderboard.

---


Run the benchmark on your embeddings with:


```bash
python main.py \
  --annotation_path path/to/annotation_folder \
  --submission_file path/to/submission_file.csv \
  --output_dir path/to/results \
  --config path/to/config.yaml \
  --method_name "your-method-name" (optional) \
  --phase "phase-name" (optional)
```

**Arguments:**  
- **`annotation_path`** — Folder containing task label files (`<task>__<type>.csv`).  
- **`submission_file`** — Path to your embeddings CSV.  
- **`output_dir`** — Destination for per-task reports, plots, and aggregated benchmark results.    
- **`config`** —  YAML file specifying cross-validation settings and logging options (see below).    
- **`method_name`** — Optional name used to tag your run. Defaults to infer from embedding csv name.  
- **`phase`** — Optional tag to group runs (e.g. `dev`, `ablation`). Defaults to `results`.  

**Output directory:**
```
output_dir/<phase>/<method_name>_<timestamp>/
```
---
## Evaluation Pipeline

NeuCo-Bench applies a task-wise linear‑probing workflow.

### 1. Data Loading & Preprocessing
- Load embeddings and task annotations
- Align and match sample IDs
- Optional: standardize embeddings
- Optional: normalize labels
- Filter to samples present in both files

### 2. Cross‑Validated Linear Probing
- Each task is evaluated independently
- Repeated shuffle‑split cross‑validation
- Train a linear model for each split
- Evaluate on the held‑out validation set

### 3. Metric Computation
- **Regression:** optimized with MSE and evaluated using R²
- **Classification:** optimized for a binary objective and evaluated with F1
- Compute mean and standard deviation across splits
- Compute Q statistic (see below)
- Optionally save plots

### 4. Result Writing & Aggregation
- Save per‑task results (JSON)
- Save run‑level summary
- Optionally aggregate all runs under the same phase into a leaderboard

---

## Configuration

A reference config is provided in `configs/sample_config.yaml`. The following options control the evaluation pipeline.

### Required Parameters
- **`batch_size`** — Batch size for linear probes
- **`epochs`** — Training epochs
- **`learning_rate`** — Optimizer learning rate
- **`k_folds`** — Number of cross‑validation folds

### Optional Parameters
- **`embedding_dim`** — Expected embedding size; smaller vectors are zero‑padded.
- **`standardize_embeddings`** — Standardize embeddings using global mean/std.
- **`normalize_labels`** — Normalize regression labels to `[0, 1]`.
- **`enable_plots`** — Save loss curves and task‑specific plots.
- **`task_filter`** — Specify tasks to evaluate (defaults to all available).
- **`update_leaderboard`** — Aggregate results across runs.
- **`output_fold_results`** — Also store per-fold metrics in the result JSON.

---

## Q-Score

To quantify per‑task stability and performance, NeuCo‑Bench reports a Q statistic:

```python
Q = mean_score / (0.02 + std_dev) * 2
```

- `mean_score` — Average performance across splits
- `std_dev` — Variability across splits

A higher Q indicates a method that is both strong and stable.

---

## Results & Leaderboard

### Per‑Task Results
Each evaluated task produces a `<task>_result.json` containing:  
- `q_stat` (see above)  
- `mean_score` (R²/F1)  
- `std_dev` (R²/F1)  

If enabled, the directory includes loss curves.

### Run‑Level Summary
Each run also generates a summary JSON that aggregates all task results for comparison.

### Aggregating Multiple Runs and Rank
You can aggregate all runs under the same phase into a leaderboard using a config run with `update_leaderboard: true` or manually running:

```python
from evaluation.results import summarize_runs
summarize_runs(output_dir=output_dir, phase=phase)
```

This produces a method‑ranking table summarizing all tasks, ranked by Q-Score.

---

## Output Structure

```
results/
└── <phase>/
    └── <method>_<timestamp>/
        ├── <task_name>/
        │   ├── <task_name>_result.json
        │   ├── loss_train.png
        │   ├── loss_validation.png
        │   └── ...
        └── run_summary.json
```

---

## Practical Notes

- Keep `embedding_dim`, `k_folds`, and seeds consistent across experiments in the same phase.
- Plots help diagnose learning instability.
- Use different `phase` names to group experiments.
- Set task filter for faster subset tests:

```yaml
task_filter:
  - cloud_cover
  - biomass_mean
```

- Disable preprocessing in case of already normalized custom data or for explicit testing of raw embedding features:

```yaml
standardize_embeddings: false
normalize_labels: false
```
