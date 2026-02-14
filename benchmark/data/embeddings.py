import json
import logging
import re
import ast
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset wrapping embeddings and targets.
    Expects a DataFrame with 'embedding' and 'label' columns.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        """PyTorch Dataset wrapping embeddings and targets.
        Expects a DataFrame with 'embedding' and 'label' columns.

        Args:
            df: Pandas dataframe with columns 'embedding' and 'label'.
        """
        self.features = torch.stack(
            [torch.tensor(e, dtype=torch.float32) for e in df['embedding']]
        )
        self.targets = torch.tensor(df['label'].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def load_submission(
    file_path: Path,
    valid_ids: Set[str],
    expected_dim: int | None = None,
    exclude_file: Optional[Path] = None,
    standardize: bool = True,
) -> (pd.DataFrame, int):
    """
    Load and preprocess CSV of embeddings.

    - Filters by IDs in valid_ids and optional exclude list.
    - Standardizes embeddings (zero mean, unit variance) if standardize is True.

    Returns a DataFrame with 'id' and 'embedding' columns.

    Args:
        file_path: Path to file containing embeddings.
        valid_ids: Set of valid embedding IDs.
        expected_dim: Expected dimension of embeddings, e.g. 1024, if None is inferred from given csv.
        exclude_file: Path to file containing embeddings to exclude from processing.
        standardize: Boolean which controls whether embeddings are standardized (default).
            Standardization is done over the complete embedding_file, not per downstream task.
    Returns:
        - Pandas Dataframe with columns 'id' and 'embedding'.
        - Embedding dimension.
    Raises:
        ValueError if embedding_file does not contain column 'id'.
        ValueError if there are more embedding dim columns than expected.
    """
    logger.info("Loading embeddings from %s", file_path)
    df = pd.read_csv(file_path)
    
    if 'id' not in df.columns:
        raise ValueError(f"""Submission file must contain column 'id'.""")

    # Strip .zarr/ or .zarr.zip from id column if present
    df["id"] = (
        df["id"]
        .astype(str)
        .str.replace(".zarr.zip", "", regex=False)
        .str.replace(".zarr", "", regex=False)
    )

    df.set_index('id', inplace=True)

    if exclude_file and exclude_file.exists():
        bad_ids = {line.strip() for line in exclude_file.read_text().splitlines() if line.strip()}
        logger.info("Excluding %d corrupted IDs", len(bad_ids))
        df = df.drop(index=bad_ids, errors='ignore')

    df = df.loc[df.index.intersection(valid_ids)]
    logger.info("Retained %d valid records", len(df))

    # Embedding dim columns are identified by column names only consisting of digits, if none are found we default to using all columns.
    digit_cols = [c for c in df.columns if str(c).isdigit()]
    if digit_cols:
        emb_cols = sorted(digit_cols, key=lambda s: int(s))
    else:
        emb_cols = list(df.columns)

    found_dim = len(emb_cols)

    # Infer embedding dimension if embedding dim is not specified
    if expected_dim is None:
        expected_dim = found_dim
        logger.info("Inferred embedding_dim=%d from submission columns.", expected_dim)
    if found_dim > expected_dim:
        raise ValueError(f"Embedding dimension mismatch. Expected {expected_dim}, got {found_dim}")

    # Replace NaNs with 0
    emb_df = df[emb_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    embeddings_array = emb_df.to_numpy(dtype=np.float32)

    # Zero-pad if embeddings have less dimensions than specified
    if found_dim < expected_dim:
        logger.info(
            "Found embedding dim %d; expected %d. Zero-padding embeddings to %d.",
            found_dim, expected_dim, expected_dim
        )
        pad = expected_dim - found_dim
        embeddings_array = np.pad(embeddings_array, ((0, 0), (0, pad)), mode="constant")

    if standardize:
        mu, sigma = embeddings_array.mean(), embeddings_array.std()
        sigma = sigma if sigma != 0 else 1
        logger.info("Standardizing embeddings (mean=%.4f, std=%.4f)", mu, sigma)
        embeddings_array = (embeddings_array - mu) / sigma

    result = pd.DataFrame({
        'id': df.index,
        'embedding': list(embeddings_array)
    }).reset_index(drop=True)

    return result, expected_dim


def parse_annotations(annotation_path: Path) -> pd.DataFrame:
    """
    Load annotations JSON and flatten into DataFrame.
    """
    raw = json.loads(annotation_path.read_text())
    rows = []
    for composite_key, items in raw.items():
        task, kind = composite_key.split("__")
        for rec in items:
            rows.append({
                'id': rec['id'],
                'label': float(rec['label']),
                'task_name': task,
                'task_type': kind,
            })
    df = pd.DataFrame(rows)
    logger.info("Loaded %d annotation entries", len(df))
    return df


__all__ = ['EmbeddingDataset', 'load_submission']
