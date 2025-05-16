import csv
import logging
from pathlib import Path
from typing import Union
import pandas as pd

logger = logging.getLogger(__name__)

def get_annotations(folder_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load annotation entries from all CSV files in a folder into a single DataFrame.

    Each CSV file should be named <task>__<type>.csv and must contain columns 'id' and 'label'.

    Args:
        folder_path: Path to the directory containing CSV annotation files.

    Returns:
        DataFrame with columns ['id', 'label', 'task_name', 'task_type'].
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Provided path is not a directory: {folder}")

    entries = []
    sorted_out = 0
    
    for csv_path in folder.glob("*.csv"):
        task_name, task_type = csv_path.stem.split("__", 1)
        with csv_path.open(newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = row.get('label')  
                if label is None or label == "":
                    sorted_out += 1
                    continue
                entries.append({
                    'id': row['id'],
                    'label': float(label),
                    'task_name': task_name,
                    'task_type': task_type,
                })
        logger.info("Processed %s: %d valid entries", csv_path.name, len(entries))

    if sorted_out:
        logger.warning("Skipped %d rows due to missing labels", sorted_out)

    df = pd.DataFrame(entries)
    logger.info("Loaded total of %d annotation entries", len(df))
    return df

__all__ = ['get_annotations']
