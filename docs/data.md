# Data

NeuCo-Bench evaluates **pre-computed embeddings on downstream tasks** and is provided together with the SSL4EO-S12-Downstream dataset, which offers a set of out-of-the-box benchmark tasks (Raw Data + Labels).

Beyond this setup, NeuCo-Bench is extendable: you can add custom tasks for your own data or contribute them back as public tasks to the benchmark.

---

## SSL4EO-S12-Downstream Dataset

SSL4EO-S12-Downstream is the reference dataset released with NeuCo-Bench and is subject to extensions with new tasks. Many of the available tasks were used in the CVPR 2025 EarthVision challenge. It provides raw data and labels ready to use with the NeuCo-Bench evaluation code.

To get started, download the data from Hugging Face:  
https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream

The dataset is structured into input imagery under `data/` and task annotations under `labels/`.

```
SSL4EO-S12-downstream/
├── data/
│   ├── s1/               # Sentinel-1 (SAR) imagery
│   │   └── part-000000   # 1000 zarr.zip files
            └── <id>.zarr.zip
│   ├── s2l1c/            # Sentinel-2 L1C (top-of-atmosphere)
│   └── s2l2a/            # Sentinel-2 L2A (surface reflectance)
├── labels/
│   ├── cloud_detection__classification.csv
│   ├── biomass_estimation__regression.csv
│   └── ... (additional tasks)
└── README.md
```

It currently contains over 13k multi-modal, multi-season datacubes (Sentinel-1, Sentinel-2 L1C, Sentinel-2 L2A) with labels for 11 regression and classification tasks. The data format aligns with [SSL4EOS12 v1.1](https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1), which is recommended as a pretraining dataset. For more details, see the [Hugging Face repository](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-downstream).  

### Loading with a Torch Dataset loader

We provide a reference PyTorch Dataset implementation: [**SSL4EODownstreamDataset**](https://github.com/embed2scale/NeuCo-Bench/blob/main/NeuCo-Bench/generate_embeddings/data/dataset.py).

It supports:

- loading the multi‑modal, multi‑season datacubes from zipped Zarr files,  
- selecting all or a subset of seasonal timestamps,  
- returning modalities either concatenated along the channel dimension or separated per modality,  
- applying custom transforms,  
- and optionally shifting Sentinel‑2 channels by +1000 to match the value range of SSL4EO‑S12 v1.1.


### Labels

Labels are provdied as single csv files per task: `<task_name>__<task_type>.csv`
- They require the columns: `id` and `label`
- Rows with missing labels are skipped
