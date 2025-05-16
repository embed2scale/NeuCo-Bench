from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)

def classification_metrics(
    y_pred: List[int],
    y_true: List[int],
) -> Dict[str, Any]:
    
    y_pred_cls = (y_pred >= 0).long()

    metrics: Dict[str, Any] = {}
    metrics["f1"] = f1_score(y_true, y_pred_cls, average='binary')
    metrics["accuracy"] = accuracy_score(y_true, y_pred_cls)
    metrics["precision"] = precision_score(
        y_true, y_pred_cls, average='binary', zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred_cls, average='binary', zero_division=0
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_cls).tolist()
    metrics["roc_auc"] = roc_auc_score(y_true, y_pred)

    return metrics


def regression_metrics(
    y_pred: List[float],
    y_true: List[float],
) -> Dict[str, float]:

    metrics = {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    return metrics
