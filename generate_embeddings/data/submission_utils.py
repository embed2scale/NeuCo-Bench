"""
See: https://github.com/DLR-MF-DAS/embed2scale-challenge-supplement/blob/main/data_loading_submission_demo/demo_load_create_submission.ipynb
"""
import pandas as pd


def create_submission_from_dict(emb_dict):
    """
    Assume dictionary has format:
    {hash-id0: embedding0, hash-id1: embedding1, ...}
    """
    df_submission = pd.DataFrame.from_dict(emb_dict, orient='index')

    # Reset index with name 'id'
    df_submission.index.name = 'id'
    df_submission.reset_index(drop=False, inplace=True)

    return df_submission


def test_submission(
    path_to_submission: str,
    expected_embedding_ids: set,
    embedding_dim: int = 1024
) -> bool:
    # Load data
    df = pd.read_csv(path_to_submission, header=0)

    # Verify that 'id' is in columns
    if 'id' not in df.columns:
        raise ValueError("Submission file must contain column 'id'.")

    # Temporarily set index to 'id'
    df.set_index('id', inplace=True)

    # Check that all samples are included
    submitted_embeddings = set(df.index)
    missing = expected_embedding_ids.difference(submitted_embeddings)
    if missing:
        n_missing = len(missing)
        raise ValueError(f"Submission is missing {n_missing} embeddings.")

    # Check that embeddings have the correct length
    if df.shape[1] != embedding_dim:
        raise ValueError(
            f"{embedding_dim} embedding dimensions expected, "
            f"but provided embeddings have {df.shape[1]} dimensions."
        )

    # Convert columns to float
    try:
        for col in df.columns:
            df[col] = df[col].astype(float)
    except Exception as e:
        raise ValueError(
            "Failed to convert embedding values to float. "
            "Check for invalid characters (e.g., empty strings, letters). "
            f"Original error: {e}"
        )

    # Check for any NaNs
    if df.isna().any().any():
        raise ValueError("Embeddings contain NaN values.")

    return True