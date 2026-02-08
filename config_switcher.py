# Configuration Switcher for Quick Testing
# Use this to easily toggle specific improvements on/off

CONFIG = {
    # Feature Engineering Switches
    'use_interactions': True,
    'use_buckets': True,
    'use_polynomial': False,

    # Model Configuration
    'depth': 6,
    'l2_leaf_reg': 3,
    'learning_rate': 0.08,
    'iterations': 2000,
    'od_wait': 100,

    # Advanced Tuning
    'random_strength': None,
    'bagging_temperature': None,
    'min_data_in_leaf': None,
    'bootstrap_type': None,
    'subsample': None,
    'auto_class_weights': None,

    # Ensemble Configuration
    'cv_folds': 5,
    'random_seeds': [0, 1, 2],

    # Validation Strategy
    'quick_eval_folds': 3,
    'n_configs_to_test': 3,

    # Runtime Management
    'use_gpu': False,
    'max_runtime_hours': 9,
    'verbose': 100,
}

PRESETS = {
    'conservative': {
        'use_interactions': True,
        'use_buckets': False,
        'use_polynomial': False,
        'depth': 5,
        'l2_leaf_reg': 5,
        'random_seeds': [0, 1, 2],
    },

    'aggressive': {
        'use_interactions': True,
        'use_buckets': True,
        'use_polynomial': True,
        'depth': 7,
        'l2_leaf_reg': 3,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        'random_seeds': [0, 1, 2, 3, 4],
    },

    'balanced': {
        'use_interactions': True,
        'use_buckets': True,
        'use_polynomial': False,
        'depth': 6,
        'l2_leaf_reg': 5,
        'random_strength': 1.0,
        'bootstrap_type': 'Bayesian',
        'bagging_temperature': 1.0,
        'random_seeds': [0, 1, 2],
    },

    'current_best': {
        'use_interactions': True,
        'use_buckets': True,
        'use_polynomial': False,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seeds': [0, 1, 2],
    },
}


def apply_preset(preset_name: str):
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from {list(PRESETS.keys())}")

    CONFIG.update(PRESETS[preset_name])
    print(f"Applied preset: {preset_name}")
    return CONFIG


def get_catboost_params(config=None):
    """Convert config to CatBoost parameters."""
    if config is None:
        config = CONFIG

    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': config['iterations'],
        'od_type': 'Iter',
        'od_wait': config['od_wait'],
        'depth': config['depth'],
        'learning_rate': config['learning_rate'],
        'task_type': 'GPU' if config['use_gpu'] else 'CPU',
        'random_seed': 42,
        'verbose': config['verbose'],
    }

    if config['l2_leaf_reg'] is not None:
        params['l2_leaf_reg'] = config['l2_leaf_reg']

    optional_params = [
        'random_strength', 'bagging_temperature', 'min_data_in_leaf',
        'bootstrap_type', 'subsample', 'auto_class_weights'
    ]

    for param in optional_params:
        if config.get(param) is not None:
            params[param] = config[param]

    return params


if __name__ == "__main__":
    print("Configuration Switcher")
    print("=" * 60)
    print("\nAvailable presets:", list(PRESETS.keys()))
    print("\nCurrent CONFIG:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Example: Apply 'balanced' preset")
    print("=" * 60)
    apply_preset('balanced')
    params = get_catboost_params()
    print("\nResulting CatBoost params:")
    for key, value in params.items():
        print(f"  {key}: {value}")
