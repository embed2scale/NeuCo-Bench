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


REQUIRED_TRAINER_PARAMETERS = ['C_start_end', 'kernel_degree_list']


def validate_config(probe_params: dict):
    """Validates that `probe_params` configurations are ok.
    
    Args:
        probe_params (dict): Configuration dictionary with probe parameters
    """

    for p in REQUIRED_TRAINER_PARAMETERS:
        assert p in probe_params, f"Parameter `{p}` missing from `probe_params`."
    
    if 'opt_params' in probe_params:
        assert 'cv' not in probe_params['opt_params'], f"Parameter `cv` must not be provided in probe_params['opt_params']."

    # check kernel_degree_list type
    assert isinstance(probe_params['kernel_degree_list'], (int, list)), f"`kernel_degree_list` must be integer or list but got {type(probe_params['kernel_degree_list'])}."
    if isinstance(probe_params['kernel_degree_list'], list):
        for d in probe_params['kernel_degree_list']:
            assert isinstance(d, int), f"Kernel degrees must be integers but got {type(d)} (kernel_degree_lilst: {probe_params['kernel_degree_list']})"

    # C_start_end checks
    assert isinstance(probe_params['C_start_end'], (list, tuple)), F"`C_start_end` must be list, but got {type(probe_params['C_start_end'])}."
    assert len(probe_params['C_start_end']) == 2, f"`C_start_end` must have exactly two elements."
    assert probe_params['C_start_end'][0] < probe_params['C_start_end'][1], f"`C_start_end` must be formatted [minimum C, maximum C] but got {probe_params['C_start_end']}."

    # kernel_coef0_start_end checks
    kernel_coef0_start_end = probe_params.get('kernel_coef0_start_end', None)
    if kernel_coef0_start_end is not None:
        assert isinstance(probe_params['kernel_coef0_start_end'], (list, tuple)), f"`kernel_coef0_start_end` must be list, but got {type(probe_params['kernel_coef0_start_end'])}."
        assert len(probe_params['kernel_coef0_start_end']) == 2, f"`kernel_coef0_start_end` must have exactly two elements."
        assert probe_params['kernel_coef0_start_end'][0] < probe_params['kernel_coef0_start_end'][1], f"`kernel_coef0_start_end` must be formatted [minimum gamma, maximum gamma] but got {type(probe_params['kernel_coef0_start_end'])}."

        # Ensure kernel degree is never 1 if we optimize for gamma.
        kernel_degree_list = probe_params['kernel_degree_list']
        if isinstance(kernel_degree_list, int):
            kernel_degree_list = [kernel_degree_list]
        assert 1 not in kernel_degree_list, f"Cannot optimize coef0 if kernel degree is 1. Exclude kernel degree 1 from kernel_degree_list ({kernel_degree_list}) or exclude input kernel_coef0_start_end."

    # kernel_gamma_start_end checks
    kernel_gamma_start_end = probe_params.get('kernel_gamma_start_end', None)
    if kernel_gamma_start_end is not None:
        assert isinstance(probe_params['kernel_gamma_start_end'], (list, tuple)), f"`kernel_gamma_start_end` must be list, but got {type(probe_params['kernel_gamma_start_end'])}."
        assert len(probe_params['kernel_gamma_start_end']) == 2, f"`kernel_gamma_start_end` must have exactly two elements."
        assert probe_params['kernel_gamma_start_end'][0] < probe_params['kernel_gamma_start_end'][1], f"`kernel_gamma_start_end` must be formatted [minimum gamma, maximum gamma] but got {type(probe_params['kernel_gamma_start_end'])}."

        # Ensure kernel degree is never 1 if we optimize for gamma.
        kernel_degree_list = probe_params['kernel_degree_list']
        if isinstance(kernel_degree_list, int):
            kernel_degree_list = [kernel_degree_list]
        assert 1 not in kernel_degree_list, f"Cannot optimize gamma if kernel degree is 1. Exclude kernel degree 1 from kernel_degree_list ({kernel_degree_list}) or exclude input kernel_gamma_start_end."


class SVMHPOptimizer:
    def __init__(
            self,
            task_type: str,
            task_name: str,
            output_dir: Path,
            filename_prefix: str,
            splitter: Callable, 
            enable_plots: bool, 
            C_start_end: list[float],
            kernel_degree_list: list[int],
            kernel_coef0_start_end: list[float] | None = None,
            kernel_gamma_start_end: list[float] | None = None,
            opt_params: dict[str, Any] | None = None,
            df: pd.DataFrame = None,
            ):
        """Train a polynomial SVM classifier or regressor using Bayesian hyperparameter optimization.

        Args:
            task_type (str): Type of the task, either 'classification', 'cls', or 'regression', 'regr'.
            task_name (str): Name of the task.
            output_dir (Path): Path to the output directory where results will be saved.
            filename_prefix (str): Prefix for the filenames used in the output directory.
            splitter (Callable): Splitter used for validation statistics, e.g. sklearn.model_selection.ShuffleSplit.
            enable_plots (bool): Whether to generate plots or not. Default is False.
            C_start_end (list[float]): List of start and end values for C hyperparameter.
            kernel_degree_list (list[int]): List of kernel degrees to evaluate. Note that `kernel_gamma_start_end` cannot be used if 1 is in `kernel_degree list`.
            kernel_coef0_start_end (list[int], optional): Start and end values for the independent term in the polynomial kernel, often denoted r in (r + xTx)^d. Defaults to not being used. Note that `kernel_coef0_start_end` cannot be used if 1 is in `kernel_degree list`.
            kernel_gamma_start_end (list[float], optional): List of start and end values for gamma hyperparameter. Defaults to not being used. Note that `kernel_gamma_start_end` cannot be used if 1 is in `kernel_degree list`.
            opt_params (dict[str, Any], optional): Keyword input arguments for skopt.BayesSearchCV.
            df (pd.DataFrame, optional): Dataframe to use for training and validation.
        """

        if opt_params is None:
            opt_params = {}

        # Validate probe parameters
        validate_config({'C_start_end': C_start_end, 
            'kernel_degree_list': kernel_degree_list, 
            'kernel_coef0_start_end': kernel_coef0_start_end,
            'kernel_gamma_start_end': kernel_gamma_start_end, 
            'opt_params': opt_params})
        
        self.task_type = task_type.lower()
        self.task_name = task_name
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.enable_plots = enable_plots
        self.splitter = splitter
        self.model = None
        self.best_state = None

        if df is not None:
            self.x = np.stack(df['embedding'].to_numpy())
            self.y = np.stack(df['label'].to_numpy())
        else:
            self.x = None
            self.y = None

        if self.task_type in ("classification", "cls"):
            self.model_type = SVC
            self.metric_fn = 'f1'
            self.scoring_fn = 'accuracy'
        elif self.task_type in ("regression", "regr"):
            self.model_type = SVR
            self.metric_fn = 'r2'
            self.scoring_fn = 'neg_mean_squared_error'
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        model = self.model_type()

        # Define parameter search space
        if isinstance(kernel_degree_list, int):
            kernel_degree_list = [kernel_degree_list]

        degree_space = Categorical(kernel_degree_list)
        self.param_space = {
                'C': Real(C_start_end[0], C_start_end[1], prior='log-uniform'),
                'degree': degree_space,
                'kernel': Categorical(['poly']),
            }
        
        if kernel_gamma_start_end is not None:
            self.param_space['gamma'] = Real(kernel_gamma_start_end[0], kernel_gamma_start_end[1], prior='log-uniform')
        
        if kernel_coef0_start_end is not None:
            self.param_space['coef0'] = Real(kernel_coef0_start_end[0], kernel_coef0_start_end[1], prior='uniform')

        
        self.opt = BayesSearchCV(
            model,
            self.param_space,
            return_train_score=True,
            scoring={'metric': self.metric_fn, 'scoring': self.scoring_fn},
            cv=splitter,
            refit='scoring',
            **opt_params,
        )

    def prepare_targets(self, y):
        """Prepare targets. 
        Classification needs targets in [-1, 1] while we typically provide in [0, 1].
        """

        y_prep = np.copy(y)
        if self.model_type == SVC:
            # Ensure targets -1 and 1
            if 0 in y_prep:
                y_prep[y_prep == 0] = -1

        return y_prep


    def fit_from_state(self, state, x=None, y=None):
        if (x is None) and (y is None):
            assert self.x is not None, f"Either training data must be provided or a dataframe should be provided during initialization but neither is provided."
            assert self.y is not None, f"Either training labels must be provided or a dataframe should be provided during initialization but neither is provided."
            x = self.x
            y = self.y

        # Ensure only single class
        assert len(y.shape) == 1, f"SVM handles single targets only. Got self.y shape {y.shape}."

        # Prepare targets
        y_svm = self.prepare_targets(y)

        self.model = self.model_type(**state)

        self.model.fit(x, y_svm)
        return self.model

    def train(self, x=None, y=None) -> None:
        """
        Fit the Bayesian hyperparameter optimization search on the training data.

        Args:
            x (np.ndarray, optional): Training features of shape (n_samples, n_features).
            y (np.ndarray, optional): Training targets of shape (n_samples,).
        """
        if (x is None) and (y is None):
            assert self.x is not None, f"Either training data must be provided or a dataframe should be provided during initialization."
            assert self.y is not None, f"Either training labels must be provided or a dataframe should be provided during initialization."
            x = self.x
            y = self.y

        # Ensure only single class
        assert len(y.shape) == 1, f"Expected y to have shape (n_targets,), but got shape {y.shape}."

        # Prepare targets
        y_svm = self.prepare_targets(y)

        # Fit with Bayesian Optimization
        self.opt.fit(x, y_svm)

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
        best_train_preds[best_train_preds == -1] = 0
        
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
            raise RuntimeError("Model has not been fitted yet. Call train() first.")

        # Save model via pickle
        with open(path / 'probe.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'model_type': self.model_type.__name__,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Write load script
        load_script = path / 'load_probe.py'
        if not load_script.exists():
            load_script.write_text(f'''# -*- coding: utf-8 -*-
"""Auto-generated, self-contained script to load the saved SVM model."""
from pathlib import Path
import pickle
import json
import numpy as np


def load_probe(checkpoint_path: Path):
    """Load a saved SVM probe.

    Args:
        checkpoint_path: Path to the directory containing probe.pkl and metadata.json.

    Returns:
        Tuple of (model, task_type).
    """
    checkpoint_path = Path(checkpoint_path)

    with open(checkpoint_path / "probe.pkl", "rb") as f:
        probe = pickle.load(f)

    with open(checkpoint_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    task_type = metadata["task_type"]
    return probe, task_type

''')
    
