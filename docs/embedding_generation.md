# Embedding Generation

NeuCo-Bench can evaluate any **1D, fixed-size embedding**. This allows comparisons across foundation model embeddings, classical compression methods, and custom neural or non-neural representations.

To evaluate embeddings, you first need to encode the input data you want to benchmark on (e.g. the SSL4EO-S12-Downstream dataset or your own custom data) and save the resulting embeddings as CSV files.

We provide example scripts in `generate_embeddings/` that illustrate the required format and include two baselines:  

- An averaging baseline across modalities  
- Downsampled embeddings from a pretrained FM (DINO ViT trained on SSL4EO)  

You can also use the [TerraTorch Embedding Generation Task](https://github.com/terrastackai/terratorch/tree/main/examples/embeddings) to export embeddings in the required CSV format by setting the output format to `neuco_csv`.

As a reference, we provide a selection of CSV files from the CVPR 2025 EarthVision challenge in the repository’s top-level `data/` directory (see `data/README.md`).

---

### Embedding Size

For size-constrained benchmarking, you can set an upper embedding size limit in the NeuCo-Bench config. For example, during the CVPR 2025 EarthVision challenge we used an embedding size of 1024. If `embedding_dim` is set in the config:  

- Embeddings larger than this will raise an error  
- Smaller embeddings are automatically zero-padded to the target size  

### Embedding Format

Embeddings must be provided as a CSV file with the following structure:

```csv
id,0,1,2,...,1023
sample_001,0.234,-0.451,0.123,...,0.342
sample_002,0.112,0.223,-0.334,...,-0.156
```

- The `id` column must match the sample IDs in the annotation files.  
  For SSL4EO-S12-Downstream, this corresponds to the file name.

- Column names for embedding dimensions are expected to be integers (`0`, `1`, `2`, …).  
  If no such columns are found, all remaining columns are used as embedding dimensions.

- IDs are automatically cleaned to remove file extensions:  
  - `scene_001.zarr.zip` → `scene_001`  
  - `scene_002.zarr` → `scene_002`

### Standardization

Embeddings can be standardized using global statistics over the full submission file:
```python
embeddings = (embeddings - mean) / std
```
which is controlled via `standardize_embeddings: true` in the `config.yaml` (recommended). Standardization is applied once over the full embedding file, not per task.

### Label Normalization

For regression tasks, labels can be normalized to the `[0, 1]` range:
```python
labels = (labels - min) / (max - min)
```
which is controlled via `normalize_labels: true` in the `config.yaml` (recommended).

### Validation and Filtering

**ID Matching**

Only samples that are present in both:
- the embeddings CSV, and
- the annotation files

are evaluated. All other samples are excluded.

**Missing Values**

- Embeddings: `NaN` values are filled with `0`
- Labels: Samples with missing labels are skipped when loading annotations
