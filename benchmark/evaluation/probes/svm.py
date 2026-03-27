import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC, SVR
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from typing import Optional, Any, Callable

from evaluation.metrics import classification_metrics
from evaluation.visualisations import plot_confusion_matrix, plot_regression_scatter
from evaluation.utils import FoldResult

class SVMHPOptimizer:
    def __init__(
            self,
            task_type: str,
            task_name: str,
            output_dir: Path,
            filename_prefix: str,
            splitter: Callable, 
            enable_plots: bool, 
            df: pd.DataFrame = None,
            C_start_end: Optional[list[float]] = None,
            kernel_degree_list: Optional[list[int]] = None,
            kernel_gamma_start_end: Optional[list[float]] = None,
            opt_params: Optional[dict[str, Any]] = None,
            ):
        """Train a polynomial SVM classifier or regressor using Bayesian hyperparameter optimization.

        Args:
            task_type (str): Type of the task, either 'classification', 'cls', or 'regression', 'regr'.
            task_name (str): Name of the task.
            output_dir (Path): Path to the output directory where results will be saved.
            filename_prefix (str): Prefix for the filenames used in the output directory.
            splitter (Callable): Splitter used for validation statistics, e.g. sklearn.model_selection.ShuffleSplit.
            enable_plots (bool): Whether to generate plots or not. Default is False.
            df (pd.DataFrame, optional): Dataframe to use for training and validation.
            C_start_end (list[float], optional): List of start and end values for C hyperparameter. Defaults to None.
            kernel_degree_list (list[int], optional): List of kernel degrees to evaluate. Defaults to [1]. Note that `kernel_gamma_start_end` cannot be used if 1 is in `kernel_degree list`.
            kernel_gamma_start_end (list[float], optional): List of start and end values for gamma hyperparameter. Defaults to [1e-6, 10]. Note that `kernel_gamma_start_end` cannot be used if 1 is in `kernel_degree list`.
            opt_params (dict[str, Any], optional): Keyword input arguments for skopt.BayesSearchCV.
        """

        self.task_type = task_type
        self.task_name = task_name
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.enable_plots = enable_plots
        self.splitter = splitter
        self.model = None
        self.best_state = None
        self.is_optimized = False

        if df is not None:
            self.x = np.stack(df['embedding'].to_numpy())
            self.y = np.stack(df['label'].to_numpy())
        else:
            self.x = None
            self.y = None

        if task_type.lower() in ("classification", "cls"):
            self.model_type = SVC
            self.metric_fn = 'f1'
            self.scoring_fn = 'neg_log_loss'
        elif task_type.lower() in ("regression", "regr"):
            self.model_type = SVR
            self.metric_fn = 'r2'
            self.scoring_fn = 'neg_mean_squared_error'
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        model = self.model_type()

        # Define parameter search space
        if C_start_end is None:
            C_start_end = [1e-6, 1e1]

        if kernel_degree_list is None:
            kernel_degree_list = [1]
        elif isinstance(kernel_degree_list, int):
            kernel_degree_list = [kernel_degree_list]

        degree_space = Categorical(kernel_degree_list)
        self.param_space = {
                'C': Real(C_start_end[0], C_start_end[1], prior='log-uniform'),
                'degree': degree_space,
                'kernel': Categorical(['poly']),
            }
        if 1 in kernel_degree_list:
            assert kernel_gamma_start_end is None, f"Cannot optimize gamma if kernel degree is one. Exclude kernel degree 1 from kernel_degree_list ({kernel_degree_list}) or input kernel_gamma_start_end."
        else:
            if kernel_gamma_start_end is None:
                kernel_gamma_start_end = [1e-6, 1e1]
            self.param_space['gamma'] = Real(kernel_gamma_start_end[0], kernel_gamma_start_end[1], prior='log-uniform')

        self.opt = BayesSearchCV(
            model,
            self.param_space,
            return_train_score=True,
            scoring={'metric': self.metric_fn, 'scoring': self.scoring_fn},
            cv=splitter,
            refit='scoring',
            **opt_params,
        )

    def fit_from_state(self, state, x=None, y=None):
        if (x is None) and (y is None):
            x = self.x
            y = self.y
        self.model = self.model_type(**state)

        # Ensure only single class
        assert len(y.shape) == 1, f"SVM handles single targets only. Got self.y shape {y.shape}."

        self.model.fit(x, y)
        return self.model

    def train(self, x=None, y=None) -> None:
        """
        Fit the Bayesian hyperparameter optimization search on the training data.

        Args:
            x (np.ndarray, optional): Training features of shape (n_samples, n_features).
            y (np.ndarray, optional): Training targets of shape (n_samples,).
        """
        if (x is None) and (y is None):
            x = self.x
            y = self.y
        # Ensure only single class
        assert len(y.shape) == 1, f"SVM handles single targets only. Got self.y shape {y.shape}."

        # Fit with Bayesian Optimization
        self.opt.fit(x, y)

        # Get best model
        self.model = self.opt.best_estimator_
        best_state = self.model.get_params(deep=True)
        self.best_model_state = best_state

        n_splits = self.splitter.get_n_splits()
        train_losses = [self.opt.cv_results_[f'split{f}_train_scoring'] for f in range(n_splits)]
        val_losses = [self.opt.cv_results_[f'split{f}_test_scoring'] for f in range(n_splits)]
        metric_values = [self.opt.cv_results_[f'split{f}_test_metric'] for f in range(n_splits)]
        best_metrics = [max(mv) for mv in metric_values]
        fold_results = [FoldResult(tl, vl, mv, bm, self.best_model_state, self.model)
                        for tl, vl, mv, bm in zip(train_losses, val_losses, metric_values, best_metrics)
                        ]
        
        best_train_preds = self.model.predict(x)
        
        if self.enable_plots:
            if self.task_type in ("classification", "cls"):
                cm = classification_metrics(
                    best_train_preds, y
                )["confusion_matrix"]
                plot_confusion_matrix(
                    np.array(cm), self.task_name,
                    self.output_dir, self.filename_prefix
                )
            else:
                plot_regression_scatter(
                    y, best_train_preds,
                    self.task_name,
                    self.output_dir, self.filename_prefix,
                )
        
        return fold_results

    def save_model(self, path: Path) -> None:
        """Save the fitted SVM model and write a Python script to load it.

        Args:
            path: Directory path where model files will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call optimize_score() first.")

        # Save model via pickle
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'model_type': self.model_type.__name__,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write load script
        load_script = path / 'load_model.py'
        if not load_script.exists():
            load_script.write_text(f'''# -*- coding: utf-8 -*-
"""Auto-generated, self-contained script to load the saved SVM model."""
from pathlib import Path
import pickle
import json
import numpy as np


def load_model(checkpoint_path: Path):
    """Load a saved SVM model.

    Args:
        checkpoint_path: Path to the directory containing model.pkl and metadata.json.

    Returns:
        Tuple of (model, task_type).
    """
    checkpoint_path = Path(checkpoint_path)

    with open(checkpoint_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    task_type = metadata["task_type"]
    return model, task_type

''')
    
