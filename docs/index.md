# NeuCo-Bench Overview

NeuCo-Bench is a **benchmarking framework** designed to evaluate how effectively **compact, fixed-size embeddings** preserve information for downstream tasks. Originally developed for the 2025 CVPR EARTHVISION Challenge, it now provides a standard, task-driven and lightweight setup for evaluating embeddings locally.

In domains like Earth Observation (EO), pipelines typically handle large volumes of multi-modal, multi-temporal image data used primarily for analytical tasks. Yet, there is no standardized, method-agnostic benchmark that evaluates fixed-size embeddings, bridging both neural compression and representation learning. NeuCo-Bench addresses this gap by evaluating embeddings directly on real-world EO tasks under explicit embedding size constraints.

NeuCo-Bench provides an initial set of EO tasks and invites community contributions of additional tasks and datasets from EO and other domains.

## Key Features

- **Model-agnostic**: Supports evaluation of any fixed-size embedding (e.g. 1024‑dim feature vectors), which enables comparison among compression and representation learning methods.
- **Task-Driven Evaluation**: Utilizes linear probes across diverse EO tasks, including land-cover proportion estimation, cloud detection, and biomass estimation. 
- **Metrics**: Incorporates signal-to-noise scores and dynamic rank aggregation to compare methods.

---

## Navigation

- **[Data](data.md)** — Dataset structure and annotation requirements.
- **[Generate Embeddings](data.md)** — Information and examples for embedding generation. 
- **[Evaluation & Results](evaluation_guide.md)** — Full evaluation pipeline, metrics, output structure and ranking method.
- **[Contributing](contributing.md)** — Add new tasks, datasets, or improvements.

---

## Workflow Overview

1. **Download data** — Get the [SSL4EO-S12-downstream dataset](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream) from Hugging Face.
2. **Create embeddings** — Use your method to generate fixed-size embeddings and save as CSV.
3. **Run evaluation** — Execute the NeuCo-Bench benchmark locally with a single command.
4. **Inspect results** — View per-task metrics, plots, and aggregated leaderboards.

---

## QuickStart

```bash
# (Optional) fresh environment
micromamba create -n neuco-bench -c conda-forge python=3.12
micromamba activate neuco-bench

# clone NeuCo-Bench
git clone https://github.com/embed2scale/NeuCo-Bench.git
cd NeuCo-Bench/benchmark
pip install -r ../requirements.txt

# run evaluation
python main.py   
--annotation_path path/to/annotation_folder   
--submission_file path/to/submission_file.csv   
--output_dir path/to/results   
--config path/to/config.yaml   
--method_name your-method-name   
```

**Arguments:**
- `annotation_path` — Directory with task label files.
- `submission_file` — Your embeddings (CSV).
- `output_dir` — Destination folder for metrics, plots, and summaries.
- `config` — YAML with CV settings and preprocessing options.
- `method_name` — Name used in result folders and leaderboard.

To force CPU execution:
```
CUDA_VISIBLE_DEVICES=""
```