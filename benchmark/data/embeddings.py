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


def unify_nans_in_embeddings(vec_str):
    """Replace missing (NaN) values in string vec_str with 0."""
    vec_str = vec_str.replace("float('nan')", "'nan'").replace('float("nan")', "'nan'")
    vec_str = re.sub(r'\\b(nan)\\b', "'nan'", vec_str, flags=re.IGNORECASE)

    try:
        vec = ast.literal_eval(vec_str)
    except Exception as e:
        raise ValueError(f"Error parsing embedding: {vec_str}, Error: {e}")

    processed_vec = []
    for item in vec:
        if isinstance(item, str) and item.lower() == "nan":
            processed_vec.append(0.0)
        else:
            val = float(item)
            processed_vec.append(0.0 if np.isnan(val) else val)

    return processed_vec


def process_embedding(embedding_str, embedding_dim):
    """Preprocess embedding to adhere to basic requirements that the vector should be embedding_dim elements long and contain no missing (NaN) values.

    Args:
        embedding_str: Embedding in format as string "[val1, val2, ...]".
        embedding_dim: Number of embedding dimensions.
    """
    if pd.isna(embedding_str):
        return [0.0] * embedding_dim

    vec = unify_nans_in_embeddings(embedding_str)

    if len(vec) != embedding_dim:
        raise ValueError(f"Embedding dimension mismatch. Expected {embedding_dim}, got {len(vec)}")

    return vec


def load_submission(
    file_path: Path,
    valid_ids: Set[str],
    expected_dim: int,
    exclude_file: Optional[Path] = None,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess CSV of embeddings.

    - Filters by IDs in valid_ids and optional exclude list.
    - Standardizes embeddings (zero mean, unit variance).

    Returns a DataFrame with 'id' and 'embedding' columns.

    Args:
        file_path: Path to file containing embeddings.
        valid_ids: Set of valid embedding IDs.
        expected_dim: Expected dimension of embeddings, e.g. 1024.
        exclude_file: Path to file containing embeddings to exclude from processing.
        standardize: Boolean which controls whether embeddings are standardized (default). Standardization is done over the complete embedding_file, not per downstream task.
    Returns:
        Pandas Dataframe with columns 'id' and 'embedding'.
    Raises:
        ValueError if embedding_file does not contain column 'id'.
        ValueError if embedding_file contains missing (NaN) values.
    """
    logger.info("Loading embeddings from %s", file_path)
    df = pd.read_csv(file_path)
    
    if 'id' not in df.columns:
        raise ValueError(f"""Submission file must contain column 'id'.""")

    df['id'] = df['id'].str.replace(".zarr.zip", "", regex=False)
    df.set_index('id', inplace=True)

    if exclude_file and exclude_file.exists():
        bad_ids = {line.strip() for line in exclude_file.read_text().splitlines() if line.strip()}
        logger.info("Excluding %d corrupted IDs", len(bad_ids))
        df = df.drop(index=bad_ids, errors='ignore')

    df = df.loc[df.index.intersection(valid_ids)]
    logger.info("Retained %d valid records", len(df))

    processed_embeddings = []
    for _, row in df.iterrows():
        embedding = process_embedding(str(list(row)), expected_dim)
        processed_embeddings.append(embedding)

    embeddings_array = np.array(processed_embeddings)

    if np.isnan(embeddings_array).any():
        raise ValueError("NaN values detected in processed embeddings.")

    if standardize:
        mu, sigma = embeddings_array.mean(), embeddings_array.std()
        sigma = sigma if sigma != 0 else 1
        logger.info("Standardizing embeddings (mean=%.4f, std=%.4f)", mu, sigma)
        embeddings_array = (embeddings_array - mu) / sigma

    result = pd.DataFrame({
        'id': df.index,
        'embedding': list(embeddings_array)
    }).reset_index(drop=True)

    return result


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
