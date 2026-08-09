# Evaluation and Results

This page describes how NeuCo-Bench evaluates embeddings, how to configure the pipeline, how results are stored, and how to aggregate runs into a leaderboard.

---

## Running the Benchmark

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

Run this from the `benchmark/` directory. If you installed the package (`pip install .`), the equivalent `neuco-bench` command is available on your `PATH` and can be run from anywhere with the same arguments:

```bash
neuco-bench --annotation_path ... --submission_file ... --config ...
```

**Required arguments:**

- **`annotation_path`** — Folder containing task label files (`<task>__<type>.csv`).
- **`submission_file`** — Path to your embeddings CSV.

**Optional arguments:**

- **`output_dir`** — Destination for per-task reports, plots, and aggregated benchmark results. Defaults to `results/`.
- **`config`** — YAML file specifying cross-validation settings and logging options (see below). Defaults to `config.yaml`.
- **`method_name`** — Name used to tag your run. Defaults to the submission file name without its extension.
- **`phase`** — Tag to group runs (e.g. `dev`, `ablation`). Defaults to `results`.
- **`exclude_file`** — Text file listing embedding IDs to drop before evaluation, one ID per line. Blank lines are ignored, and IDs that are not present in the submission are ignored. Useful for excluding known-corrupted samples.

**Output directory:**

```text
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
- Each split contributes its best validation metric
- Compute mean and standard deviation across splits
- Compute Q statistic (see below)
- Optionally save plots

### 4. Result Writing & Aggregation
- Save per‑task results (JSON)
- Save run‑level summary
- Optionally aggregate all runs under the same phase into a leaderboard

---

## Configuration

Reference configs are provided in `configs/`. The following options control the evaluation pipeline.

### Required Parameters
- **`k_folds`** — Number of ShuffleSplit folds for score aggregation. Each fold holds out 10% of the samples for validation.
- **`probe_type`** — Type of probe to use. Either `linear` or `svm`.
- **`probe_params`** - Key-value parameters for the probe. See below for details.
- **`update_leaderboard`** — Aggregate results across all runs in the phase after evaluating. Must be present, otherwise the run fails after evaluation.

### Optional Parameters
- **`embedding_dim`** — Expected embedding size; smaller vectors are zero‑padded, larger ones raise an error. Default is to infer from data.
- **`standardize_embeddings`** — Standardize embeddings using global mean/std. Default is `true`.
- **`normalize_labels`** — Normalize regression labels to `[0, 1]`. Default is `true`.
- **`enable_plots`** — Save loss curves and task‑specific plots. Default is `true` (both sample configs set it to `false`).
- **`task_filter`** — Specify tasks to evaluate, as a list of task names (defaults to all available).
- **`output_fold_results`** — Also store per-fold metrics in the result JSON, under the key `q_t`. Default is False.
- **`store_probes`** — Store probe from each task. Default is False. Retrains the model on all task data, using the best hyperparameters if HP optimization is done, before storing the model. Stored under `probe` in the result folder.

#### Probe parameters
- **`probe_params`** (**`probe_type = linear`**)
  - **`batch_size`** (Required) — Batch size for linear probes.
  - **`epochs`** (Required) — Number of epochs.
  - **`learning_rate`** (Required) — Optimizer learning rate.
  - **`device`** (Optional) — Device to train probe on. Either `gpu`, `cpu` or `auto`. Default is "auto", which picks GPU if it exists and otherwise CPU.
- **`probe_params`** (**`probe_type = svm`**)
  - **`C_start_end`** (Required) — List of min, max values for C parameter in hyperparameter optimization.
  - **`kernel_degree_list`** (Required) — Integer or list of values for degree parameter. Note that **`kernel_coef0_start_end`** nor **`kernel_gamma_start_end`** cannot be used if 1 is in **`kernel_degree_list`** as it messes with the optimization.
  - **`kernel_coef0_start_end`** (optional) List of min, max values for coef0 parameter in hyperparameter optimization. Note that **`kernel_coef0_start_end`** cannot be used if 1 is in **`kernel_degree_list`**, see **`kernel_degree_list`**.
  - **`kernel_gamma_start_end`** (Optional) — List of min, max values for gamma parameter. Note that **`kernel_gamma_start_end`** cannot be used if 1 is in **`kernel_degree_list`**, see **`kernel_degree_list`**.
  - **`epsilon_list`** (Optional) — List of values for epsilon hyperparameter in Suppoert vector regression problems. Ignored for classification tasks.
  - **`opt_params`** (Optional) — Dictionary with key-value arguments for BayesSearchCV. Do not use parameter `cv`, as this is controlled by NeuCo-Bench.
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
- `hyperparamaters` — the probe hyperparameters used (note the spelling of this key)
- `q_t` — the list of per-fold scores, only when `output_fold_results: true`

If enabled, the directory includes loss curves.

### Run‑Level Summary
Each run also generates `results_summary.json`, which aggregates all task results for comparison. It contains `overall_q_score` and `overall_acc_score` (means across tasks), the per-task `task_q_scores` and `task_acc_scores`, and a snapshot of the config used.

### Aggregating Multiple Runs and Rank
You can aggregate all runs under the same phase into a leaderboard using a config run with `update_leaderboard: true` or manually running:

```python
from evaluation.results import summarize_runs
summarize_runs(output_dir=output_dir, phase=phase)
```

This scans every `results_summary.json` under the phase and produces a method‑ranking table, saved to `<output_dir>/leaderboards/<phase>/leaderboard_<timestamp>.csv`.

Ranking uses stddev‑weighted rank aggregation over the per‑task Q‑Scores rather than a plain average:

1. Runs missing a Q‑Score for **any** task are dropped from the ranking.
2. For each task, methods are ranked by Q‑Score (highest → rank 1).
3. Each task is weighted by the standard deviation of its scores across methods, divided by the sum of all such standard deviations — so tasks that discriminate between methods count more.
4. `weighted_score` is the sum of `rank × weight`, and `aggregated_rank` ranks methods by it (lowest is best).

!!! warning
    Step 1 is silent. If one method fails or skips a task, that entire run disappears from the leaderboard. Keep `task_filter` identical across all runs you intend to compare in the same phase.

---

## Output Structure

```text
results/                                  # output_dir
├── <phase>/
│   └── <method>_<timestamp>/
│       ├── <task_name>/
│       │   ├── <task_name>_result.json
│       │   ├── loss_curves/              # always written
│       │   │   ├── <task_name>_train_loss_curves.png
│       │   │   └── <task_name>_validation_loss_curves.png
│       │   ├── regression_scatter/       # enable_plots, regression tasks
│       │   ├── confusion_matrices/       # enable_plots, classification tasks
│       │   └── probe/                    # store_probes: true
│       │       ├── probe.pth              # linear probe (or probe.pkl
│       │       │                          # + metadata.json for svm)
│       │       └── load_probe.py          # generated loader
│       └── results_summary.json
└── leaderboards/                         # update_leaderboard: true
    └── <phase>/
        └── leaderboard_<timestamp>.csv
```

Note that `leaderboards/` sits directly under `output_dir`, alongside `<phase>/` rather than inside the individual run folders.

---

## Practical Notes

- Keep `embedding_dim`, `k_folds`, and seeds consistent across experiments in the same phase.
- Plots help diagnose learning instability.
- Use different `phase` names to group experiments.
- Set task filter for faster subset tests. Use the task name only — the part of the label filename before `__`, without the task type:

```yaml
task_filter:
  - biomass_mean
  - biomass_std
```

- Disable preprocessing in case of already normalized custom data or for explicit testing of raw embedding features:

```yaml
standardize_embeddings: false
normalize_labels: false
```

### Working with stored probes

The stored probes are provided with a self-contained load script to simplify use. The same `load_probe(probe_folder)` call works for both probe types — it returns a `(probe, task_type)` tuple regardless of whether the underlying file is a linear `probe.pth` or an SVM `probe.pkl`.

Below is a minimal example to access the trained probe:
```python
import importlib
from pathlib import Path
import sys

probe_folder = Path("/path/to/task/results/phase_name/method_name/task_name/probe/")

# Import the load_probe functionality
spec=importlib.util.spec_from_file_location("load_probe", probe_folder / "load_probe.py")

# creates a new module based on spec
load_probe = importlib.util.module_from_spec(spec, )

# instantiate module
sys.modules[spec.name] = load_probe 
spec.loader.exec_module(load_probe)

# Load trained probe
probe, task_type = load_probe.load_probe(probe_folder)
```