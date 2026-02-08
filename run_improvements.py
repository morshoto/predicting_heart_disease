# Heart Disease Competition - Improvement Pipeline
# Goal: Improve from 0.95366 to 0.95376+ (delta +0.0001 to +0.0003)

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from heart_disease_improvements import (
    add_interaction_features,
    add_bucketed_features,
    add_polynomial_features,
    run_catboost_cv_fast,
    run_catboost_cv_full,
    train_and_predict_seeds,
    get_search_configs,
    compute_quantile_bins,
)


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    kaggle_dir = Path("/kaggle/input/playground-series-s6e2")
    if kaggle_dir.exists():
        train = pd.read_csv(kaggle_dir / "train.csv")
        test = pd.read_csv(kaggle_dir / "test.csv")
        sub = pd.read_csv(kaggle_dir / "sample_submission.csv")
        output_dir = Path("/kaggle/working")
        return train, test, sub, output_dir

    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "raw"
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sub_path = data_dir / "sample_submission.csv"

    if train_path.exists() and test_path.exists() and sub_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        sub = pd.read_csv(sub_path)
        output_dir = root / "data" / "submissions"
        return train, test, sub, output_dir

    # Fallback to current working directory files
    cwd = Path.cwd()
    if (cwd / "train.csv").exists() and (cwd / "test.csv").exists() and (cwd / "sample_submission.csv").exists():
        train = pd.read_csv(cwd / "train.csv")
        test = pd.read_csv(cwd / "test.csv")
        sub = pd.read_csv(cwd / "sample_submission.csv")
        output_dir = cwd
        return train, test, sub, output_dir

    raise FileNotFoundError(
        "Could not find train/test/sample_submission. "
        "Expected in /kaggle/input/playground-series-s6e2, data/raw, or current directory."
    )


# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train, test, sub, OUTPUT_DIR = _load_data()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# =============================================================================
# 2. PREPARE BASE DATA
# =============================================================================

print("\n" + "=" * 80)
print("PREPARING BASE DATA")
print("=" * 80)

TARGET_RAW = "Heart Disease"
target_map = {"Absence": 0, "Presence": 1}
y = train[TARGET_RAW].map(target_map)
print(f"Positive rate: {y.mean():.4f}")

ID_COL = "id"

CAT_COLS_BASE = [
    "Sex",
    "FBS over 120",
    "Exercise angina",
    "EKG results",
    "Slope of ST",
    "Thallium",
    "Number of vessels fluro",
    "Chest pain type",
]

feature_cols = [c for c in train.columns if c not in [TARGET_RAW, ID_COL]]

X_train_base = train[feature_cols].copy()
X_test_base = test[feature_cols].copy()

# Zero indicators
for c in ["ST depression", "Number of vessels fluro"]:
    X_train_base[f"{c}__is_zero"] = (X_train_base[c] == 0).astype("int8")
    X_test_base[f"{c}__is_zero"] = (X_test_base[c] == 0).astype("int8")

print(f"Base features: {X_train_base.shape[1]}")

# Precompute quantile bins for Max HR for consistency across train/test
maxhr_bins = None
if "Max HR" in X_train_base.columns:
    maxhr_bins = compute_quantile_bins(X_train_base["Max HR"], q=5)

# =============================================================================
# 3. FEATURE ENGINEERING EXPERIMENTS
# =============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

feature_sets = {}

# Feature Set 1: Baseline
X_train_fs1 = X_train_base.copy()
X_test_fs1 = X_test_base.copy()
cat_cols_fs1 = CAT_COLS_BASE.copy()

for c in cat_cols_fs1:
    X_train_fs1[c] = X_train_fs1[c].astype(str)
    X_test_fs1[c] = X_test_fs1[c].astype(str)

feature_sets["baseline"] = {
    "X_train": X_train_fs1,
    "X_test": X_test_fs1,
    "cat_cols": cat_cols_fs1,
    "description": "Baseline features with zero indicators",
}

# Feature Set 2: + Interactions
X_train_fs2 = X_train_base.copy()
X_test_fs2 = X_test_base.copy()
cat_cols_fs2 = CAT_COLS_BASE.copy()

X_train_fs2, cat_cols_fs2 = add_interaction_features(X_train_fs2, cat_cols_fs2)
X_test_fs2, _ = add_interaction_features(X_test_fs2, CAT_COLS_BASE.copy())

cat_cols_fs2 = list(dict.fromkeys(cat_cols_fs2))
for c in cat_cols_fs2:
    if c in X_train_fs2.columns:
        X_train_fs2[c] = X_train_fs2[c].astype(str)
        X_test_fs2[c] = X_test_fs2[c].astype(str)

feature_sets["interactions"] = {
    "X_train": X_train_fs2,
    "X_test": X_test_fs2,
    "cat_cols": cat_cols_fs2,
    "description": "Baseline + interaction features",
}

print(f"Interactions feature count: {X_train_fs2.shape[1]}")

# Feature Set 3: + Buckets
X_train_fs3 = X_train_base.copy()
X_test_fs3 = X_test_base.copy()
cat_cols_fs3 = CAT_COLS_BASE.copy()

X_train_fs3, cat_cols_fs3, _ = add_bucketed_features(X_train_fs3, cat_cols_fs3, maxhr_bins=maxhr_bins)
X_test_fs3, _, _ = add_bucketed_features(X_test_fs3, CAT_COLS_BASE.copy(), maxhr_bins=maxhr_bins)

cat_cols_fs3 = list(dict.fromkeys(cat_cols_fs3))
for c in cat_cols_fs3:
    if c in X_train_fs3.columns:
        X_train_fs3[c] = X_train_fs3[c].astype(str)
        X_test_fs3[c] = X_test_fs3[c].astype(str)

feature_sets["buckets"] = {
    "X_train": X_train_fs3,
    "X_test": X_test_fs3,
    "cat_cols": cat_cols_fs3,
    "description": "Baseline + bucketed features",
}

print(f"Buckets feature count: {X_train_fs3.shape[1]}")

# Feature Set 4: + Interactions + Buckets
X_train_fs4 = X_train_base.copy()
X_test_fs4 = X_test_base.copy()
cat_cols_fs4 = CAT_COLS_BASE.copy()

X_train_fs4, cat_cols_fs4 = add_interaction_features(X_train_fs4, cat_cols_fs4)
X_test_fs4, _ = add_interaction_features(X_test_fs4, CAT_COLS_BASE.copy())

X_train_fs4, cat_cols_fs4, _ = add_bucketed_features(X_train_fs4, cat_cols_fs4, maxhr_bins=maxhr_bins)
X_test_fs4, _, _ = add_bucketed_features(X_test_fs4, CAT_COLS_BASE.copy(), maxhr_bins=maxhr_bins)

cat_cols_fs4 = list(dict.fromkeys(cat_cols_fs4))
for c in cat_cols_fs4:
    if c in X_train_fs4.columns:
        X_train_fs4[c] = X_train_fs4[c].astype(str)
        X_test_fs4[c] = X_test_fs4[c].astype(str)

feature_sets["combined"] = {
    "X_train": X_train_fs4,
    "X_test": X_test_fs4,
    "cat_cols": cat_cols_fs4,
    "description": "Baseline + interactions + buckets",
}

print(f"Combined feature count: {X_train_fs4.shape[1]}")

# Feature Set 5: Combined + Polynomial
X_train_fs5 = add_polynomial_features(X_train_fs4)
X_test_fs5 = add_polynomial_features(X_test_fs4)
cat_cols_fs5 = cat_cols_fs4.copy()

for c in cat_cols_fs5:
    if c in X_train_fs5.columns:
        X_train_fs5[c] = X_train_fs5[c].astype(str)
        X_test_fs5[c] = X_test_fs5[c].astype(str)

feature_sets["combined_poly"] = {
    "X_train": X_train_fs5,
    "X_test": X_test_fs5,
    "cat_cols": cat_cols_fs5,
    "description": "Combined + polynomial features",
}

print(f"Combined+Poly feature count: {X_train_fs5.shape[1]}")

# =============================================================================
# 4. QUICK FEATURE SET EVALUATION (3-FOLD)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1: QUICK FEATURE SET EVALUATION (3-FOLD)")
print("=" * 80)

base_params_quick = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 1500,
    'od_type': 'Iter',
    'od_wait': 80,
    'depth': 6,
    'learning_rate': 0.08,
    'task_type': 'CPU',
    'random_seed': 42,
    'verbose': False,
}

fs_results = []

for fs_name, fs_data in feature_sets.items():
    print(f"\nTesting feature set: {fs_name}")
    print(f"  Description: {fs_data['description']}")
    print(f"  Features: {fs_data['X_train'].shape[1]}")

    start_time = time.time()
    mean_auc, std_auc = run_catboost_cv_fast(
        fs_data['X_train'], y, base_params_quick,
        fs_data['cat_cols'], n_splits=3, random_state=42, tag=fs_name
    )
    elapsed = time.time() - start_time

    fs_results.append({
        'feature_set': fs_name,
        'n_features': fs_data['X_train'].shape[1],
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'time_3fold_sec': elapsed,
    })

    print(f"  3-fold CV AUC: {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"  Time: {elapsed:.1f}s")

fs_results_df = pd.DataFrame(fs_results).sort_values('mean_auc', ascending=False)
print("\n" + "=" * 60)
print("FEATURE SET COMPARISON")
print("=" * 60)
print(fs_results_df.to_string(index=False))

best_fs_name = fs_results_df.iloc[0]['feature_set']
best_fs_data = feature_sets[best_fs_name]
print(f"\nBest feature set: {best_fs_name}")
print(f"  AUC: {fs_results_df.iloc[0]['mean_auc']:.6f}")
print(f"  Features: {fs_results_df.iloc[0]['n_features']}")

# =============================================================================
# 5. HYPERPARAMETER SEARCH (3-FOLD)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 2: HYPERPARAMETER SEARCH (3-FOLD)")
print("=" * 80)

configs = get_search_configs()
hp_results = []

for config in configs:
    print(f"\nTesting config: {config['name']}")

    start_time = time.time()
    mean_auc, std_auc = run_catboost_cv_fast(
        best_fs_data['X_train'], y, config['params'],
        best_fs_data['cat_cols'], n_splits=3, random_state=42,
        tag=config['name']
    )
    elapsed = time.time() - start_time

    hp_results.append({
        'config': config['name'],
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'time_3fold_sec': elapsed,
        'params': config['params'],
    })

    print(f"  3-fold CV AUC: {mean_auc:.6f} ± {std_auc:.6f}")
    print(f"  Time: {elapsed:.1f}s")

hp_results_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != 'params'} for r in hp_results]
).sort_values('mean_auc', ascending=False)
print("\n" + "=" * 60)
print("HYPERPARAMETER COMPARISON (TOP 5)")
print("=" * 60)
print(hp_results_df.head().to_string(index=False))

# Select top 3 configs for full validation
top_k = min(3, len(hp_results_df))
top_config_names = hp_results_df.head(top_k)['config'].tolist()
print(f"\nTop {top_k} configs for 5-fold validation: {top_config_names}")

# =============================================================================
# 6. FULL VALIDATION (5-FOLD) OF TOP CONFIGS
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 3: FULL VALIDATION (5-FOLD)")
print("=" * 80)

full_validation_results = []

for config_dict in hp_results:
    if config_dict['config'] in top_config_names:
        print(f"\n{'=' * 60}")
        print(f"Full 5-fold validation: {config_dict['config']}")
        print(f"{'=' * 60}")

        cv_table = run_catboost_cv_full(
            best_fs_data['X_train'], y, config_dict['params'],
            best_fs_data['cat_cols'], n_splits=5, random_state=42,
            tag=config_dict['config']
        )

        mean_auc = cv_table['roc_auc'].mean()
        std_auc = cv_table['roc_auc'].std(ddof=1)
        median_best_it = int(cv_table['best_iteration'].median())
        final_iterations = int(median_best_it * 1.15)

        full_validation_results.append({
            'config': config_dict['config'],
            'mean_auc_5fold': mean_auc,
            'std_auc_5fold': std_auc,
            'median_best_iteration': median_best_it,
            'final_iterations': final_iterations,
            'params': config_dict['params'],
        })

        print(f"\n5-fold CV AUC: {mean_auc:.6f} ± {std_auc:.6f}")
        print(f"Median best iteration: {median_best_it}")
        print(f"Recommended final iterations: {final_iterations}")

full_val_df = pd.DataFrame(
    [{k: v for k, v in r.items() if k != 'params'} for r in full_validation_results]
).sort_values('mean_auc_5fold', ascending=False)

print("\n" + "=" * 60)
print("FULL VALIDATION RESULTS")
print("=" * 60)
print(full_val_df.to_string(index=False))

best_config_dict = full_validation_results[0]
best_config_name = best_config_dict['config']
best_params = best_config_dict['params'].copy()
best_final_iterations = best_config_dict['final_iterations']

print(f"\n{'=' * 60}")
print(f"BEST CONFIGURATION: {best_config_name}")
print(f"{'=' * 60}")
print(f"5-fold CV AUC: {best_config_dict['mean_auc_5fold']:.6f} ± {best_config_dict['std_auc_5fold']:.6f}")
print(f"Final iterations: {best_final_iterations}")

best_params['iterations'] = best_final_iterations
best_params['verbose'] = False

# =============================================================================
# 7. FINAL TRAINING WITH SEED ENSEMBLE
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 4: FINAL TRAINING WITH SEED ENSEMBLE")
print("=" * 80)

SEEDS_INITIAL = [0, 1, 2]
SEEDS_EXTENDED = [0, 1, 2, 3, 4]

print(f"Feature set: {best_fs_name}")
print(f"Config: {best_config_name}")
print(f"Starting with seeds: {SEEDS_INITIAL}")

start_final_time = time.time()

test_pred, oof_pred, oof_auc = train_and_predict_seeds(
    best_fs_data['X_train'], y, best_fs_data['X_test'],
    best_params, best_fs_data['cat_cols'],
    n_splits=5, seeds=SEEDS_INITIAL
)

elapsed_final = time.time() - start_final_time
print(f"\nTime for {len(SEEDS_INITIAL)} seeds: {elapsed_final/60:.1f} minutes")

RUNTIME_LIMIT_HOURS = 9
estimated_total_time = elapsed_final / len(SEEDS_INITIAL) * len(SEEDS_EXTENDED)

time_remaining = RUNTIME_LIMIT_HOURS * 3600 - (time.time() - start_final_time)

if time_remaining > estimated_total_time * 1.2:
    print(f"\nTime remaining: {time_remaining/3600:.1f}h")
    print(f"Expanding to {len(SEEDS_EXTENDED)} seeds...")

    test_pred, oof_pred, oof_auc = train_and_predict_seeds(
        best_fs_data['X_train'], y, best_fs_data['X_test'],
        best_params, best_fs_data['cat_cols'],
        n_splits=5, seeds=SEEDS_EXTENDED
    )
else:
    print(f"\nTime remaining: {time_remaining/3600:.1f}h")
    print(f"Sticking with {len(SEEDS_INITIAL)} seeds to stay within budget")

# =============================================================================
# 8. CREATE SUBMISSION
# =============================================================================

print("\n" + "=" * 80)
print("CREATING SUBMISSION")
print("=" * 80)

submission = sub.copy()
submission["Heart Disease"] = test_pred

submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Final OOF AUC: {oof_auc:.6f}")
print("\nSubmission preview:")
print(submission.head(10))

# =============================================================================
# 9. SUMMARY REPORT
# =============================================================================

print("\n" + "=" * 80)
print("IMPROVEMENT SUMMARY REPORT")
print("=" * 80)

summary = f"""
BASELINE PERFORMANCE:
- Public LB: 0.95366 (V3)

IMPROVEMENTS APPLIED:
1. Feature Engineering:
   - Selected feature set: {best_fs_name}
   - Total features: {best_fs_data['X_train'].shape[1]}
   - Baseline features: {X_train_base.shape[1]}

2. Hyperparameter Tuning:
   - Best config: {best_config_name}
   - Final iterations: {best_final_iterations}

3. Ensemble Strategy:
   - Random seeds: {SEEDS_INITIAL if time_remaining <= estimated_total_time * 1.2 else SEEDS_EXTENDED}
   - CV folds: 5

FINAL PERFORMANCE:
- OOF AUC: {oof_auc:.6f}
- Expected LB: ~{oof_auc:.5f} (conservative estimate)

EXPECTED IMPROVEMENT:
- Delta from baseline: +{oof_auc - 0.95366:.6f}
- Target was: +0.0001 to +0.0003
- Status: {'✓ TARGET MET' if oof_auc - 0.95366 >= 0.0001 else '✗ BELOW TARGET'}
"""

print(summary)

summary_path = OUTPUT_DIR / "improvement_summary.txt"
with open(summary_path, "w") as f:
    f.write(summary)

print(f"\nSummary saved to: {summary_path}")
print("\n" + "=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
