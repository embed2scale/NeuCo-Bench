# NeuCo-Bench

**Licence**: Apache-2.0

*Originally developed to evaluate challenge submissions in the 2025 EarthVision Challenge at CVPR—see the [competition details](https://www.grss-ieee.org/events/earthvision-2025/?tab=challenge)—NeuCo-Bench is now released for local benchmarking and evaluation.*

NeuCo-Bench is a **benchmarking framework** for evaluating how well compressed embeddings preserve the information needed for downstream tasks.
In domains like Earth Observation (EO), image data is mostly used for analysis tasks but pipelines suffer from large data volumes. Traditional compression focusses on pixel-level reconstruction, while Foundation Model (FM) research does not explicitly consider embedding size. NeuCo-Bench fills this gap by enforcing strict size constraints and testing embeddings directly on real-world EO tasks.

NeuCo-Bench includes an initial set of EO tasks and welcomes the contribution of additional tasks and datasets from EO and other domains.

<p align="center">
  <img src="assets/NeuCoBench.png" alt="Framework overview" width="100%" />
</p>


## Key Features

- **Model-agnostic**: Accepts any fixed-size embedding (e.g., 1,024‑dimensional feature vector), which enables comparison of compression and representation learning methods.
- **Task-Driven Evaluation**: Applies linear probes across diverse EO task such as land-cover proportion, cloud detection, and biomass estimation. 
- **Metrics**: Integrates signal-to-noise scores and dynamic rank aggregation to compare methods.

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

- `--annotation_path` Directory with CSV label files for each task.  
- `--submission_file` CSV file containing your embeddings.  
- `--output_dir` Destination for per-task reports, plots, and aggregated benchmark results.  
- `--config` YAML file specifying cross-validation settings and logging options. See provided sample config.  
- `--method_name` Identifier for your method, used in filenames and leaderboard entries.  
- `--phase` A name that groups a set of evaluation runs for joint ranking. Results for each phase are stored in a separate subfolder under `output_dir`. 

If you'd like to avoid utilization of GPUs, run `CUDA_VISIBLE_DEVICES=''` before execution.

## Overview

NeuCo-Bench moves beyond pixel-level reconstruction to task-oriented semantic evaluation, and measures how well compressed embeddings preserve information for EO tasks.

To evaluate a method with NeuCo-Bench, the workflow is:
1. Download the SSL4EO-S12-downstream data (images + labels) from Hugging Face. 
2. Encode imagees into fixed-size embeddings and save as CSV file.
3. Run NeuCo-Bench locally to evaluate each method and aggregate scores across methods to build a leaderboard.


**1. Data**

We provide a set of donwstream tasks in [SSL4EO-S12-downstream on Hugging Face](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream). 
It is preprocessed and organized in the same format as [SSL4EOS12 v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1), which we therefore recommend using as a pretraining dataset for NeuCo-Bench.


The downstream dataset consists of two folders:

- `data/`  
  Contains three modality subfolders—`s1/`, `s2l1c/`, and `s2l2a/`—each split into parts, with 1,000 `zarr.zip` files.

- `labels/`  
  Holds annotation files for each downstream task.


Both `data/` and `labels/` are required. See the `examples/` on how to load the data as a torch dataset. If you experience data-loading errors, verify that `zarr==2.18.0` is used.



**2. Creating Embeddings**

Use your preferred method to generate fixed-size embeddings and save them as a CSV file. Example scripts in `examples/` show the expected CSV format. For comparison, all methods should use the same embedding dimension—for instance, we set a 1,024-dim size limit during the CVPR EarthVision Challenge.



**3. Evaluation and Ranking**

After generating embeddings, evaluate an embedding csv file with:

```bash
python main.py \
  --annotation_path path/to/annotation_folder \
  --submission_file path/to/submission_file.csv \
  --output_dir path/to/results \
  --config path/to/config.yaml \
  --method_name "your-method-name" \
  --phase "phase-name"
```

Config File: A default setting can be seen in sample `config`. Adjustable parameters include:  

- `batch_size`, `epochs`, `learning_rate`, `k_folds`: Settings for Cross-Validation.  
- `standardize_embeddings`: Standardize embeddings using global mean and std before evaluation (Recommended).  
- `normalize_labels`: Normalize target labels to [0,1] before evaluation (Recommended).  
- `enable_plots`: Generate per-fold plots (e.g., parity plots for regression).  
- `update_leaderboard`: Aggregate and update leaderboard after evaluation.  
- `task_filter`: List of task names to include (defaults to all available in `annotation_path`). 

Results appear under `output_dir/phase_name/` containing 
- Per-task metric insights and loss curves
- A `results_summary.json` with per-task and overall signal-to-noise scores


To aggregate and rank all runs in a phase, run the following:

```bash
from evaluation.results import summarize_runs
summarize_runs(output_dir=output_dir, phase=phase)
```

---

## Future Work & Contributing

All downstream tasks and labels are published on HuggingFace. We are planning to extend the framework to further tasks (eg. spatial and temporal downstream tasks).

We invite the community to collaborate and appreciate contributions, including but not limited to the following:
- Benchmark and contribute new compression techniques
- Incorporate additional downstream task and metrics
- Extension to further input modalities

Check out [CONTRIBUTING.md](.github/CONTRIBUTING.md).
