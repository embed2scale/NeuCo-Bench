
from evaluation.probes.linear import validate_config as linear_validate_config
from evaluation.probes.svm import validate_config as svm_validate_config

def check_config(config: dict):
    """Validates that the supplied configuration file is ok.
    
    Args:
        Config (dict): Configuration dictionary
    """

    probe_info = {'linear': {'val_fun': linear_validate_config}, 
                  'svm': {'val_fun': svm_validate_config}
                  }

    # Base parameters
    for param in ['k_folds', 'probe_type', 'probe_params']:
        assert param in config.keys(), f"Parameter `{param}` missing from  config."

    # Optional parameters
    if 'embedding_dim' in config.keys():
        assert (config['embedding_dim'] is None) or (isinstance(config['embedding_dim'], int)), f"`embedding_dim` must be integer or None but got {type(config['embedding_dim'])}."

    bool_params = ['standardize_embeddings', 'normalize_labels', 'enable_plots', 'update_leaderboard', 'output_fold_results', 'store_probes']
    for p in bool_params:
        if p in config.keys():
            assert (isinstance(config[p], bool)), f"`{p}` must be boolean but got {type(config[p])}."
    
    if 'task_filter' in config.keys():
        assert (config['task_filter'] is None) or (isinstance(config['task_filter'], list)), f"`task_filter` must be list or None but got {type(config['task_filter'])}."
        if config['task_filter'] is not None:
            for t in config['task_filter']:
                assert isinstance(t, str), f"`task_filter elements must be strings but got {type(t)} for task {t} (task_filter: {config['task_filter']})."

    # Probe parameters
    if config['probe_type'] in probe_info.keys():
        probe_info[config['probe_type']]['val_fun'](config['probe_params'])
    else:
        raise NotImplementedError(f"probe_type `{config['probe_type']}` is not implemented. Currently implemented probes are: {list(probe_info.keys())}")
    
    
        

