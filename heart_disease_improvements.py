"""
Heart Disease Kaggle Competition - Improvement Utilities
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def compute_quantile_bins(series: pd.Series, q: int = 5) -> np.ndarray:
    """Compute stable quantile bin edges for pd.cut."""
    quantiles = np.linspace(0.0, 1.0, q + 1)
    edges = np.unique(np.quantile(series.dropna().values, quantiles))
    if len(edges) < 2:
        # Fallback to min/max bounds
        min_v = float(series.min())
        max_v = float(series.max())
        edges = np.array([min_v - 1e-6, max_v + 1e-6])
    edges[0] = edges[0] - 1e-6
    edges[-1] = edges[-1] + 1e-6
    return edges


def add_interaction_features(df: pd.DataFrame, cat_cols_model: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Add high-value interaction features."""
    df = df.copy()
    cat_cols_model = list(cat_cols_model)

    if 'Age' in df.columns and 'Sex' in df.columns:
        df['Age_X_Sex'] = df['Age'].astype(str) + '_' + df['Sex'].astype(str)
        cat_cols_model.append('Age_X_Sex')

    if 'Max HR' in df.columns and 'Age' in df.columns:
        df['MaxHR_div_Age'] = (df['Max HR'] / (df['Age'] + 1)).round(2)
        df['MaxHR_vs_Expected'] = df['Max HR'] - (220 - df['Age'])

    if 'Age' in df.columns and 'Cholesterol' in df.columns:
        df['Age_X_Cholesterol'] = (df['Age'] * df['Cholesterol']).astype(float)

    if 'ST depression' in df.columns and 'Slope of ST' in df.columns:
        df['STdep_X_Slope'] = df['ST depression'].round(1).astype(str) + '_' + df['Slope of ST'].astype(str)
        cat_cols_model.append('STdep_X_Slope')

    if 'Thallium' in df.columns and 'Chest pain type' in df.columns:
        df['Thallium_X_ChestPain'] = df['Thallium'].astype(str) + '_' + df['Chest pain type'].astype(str)
        cat_cols_model.append('Thallium_X_ChestPain')

    if 'Number of vessels fluro' in df.columns and 'Thallium' in df.columns:
        df['Vessels_X_Thallium'] = df['Number of vessels fluro'].astype(str) + '_' + df['Thallium'].astype(str)
        cat_cols_model.append('Vessels_X_Thallium')

    if 'EKG results' in df.columns and 'Exercise angina' in df.columns:
        df['EKG_X_Angina'] = df['EKG results'].astype(str) + '_' + df['Exercise angina'].astype(str)
        cat_cols_model.append('EKG_X_Angina')

    if 'Cholesterol' in df.columns and 'Age' in df.columns:
        age_bracket = pd.cut(
            df['Age'],
            bins=[0, 40, 50, 60, 70, 100],
            labels=['<40', '40-50', '50-60', '60-70', '70+'],
        )
        df['Chol_X_AgeBracket'] = df['Cholesterol'].round(-1).astype(str) + '_' + age_bracket.astype(str)
        cat_cols_model.append('Chol_X_AgeBracket')

    return df, cat_cols_model


def add_bucketed_features(
    df: pd.DataFrame,
    cat_cols_model: list[str],
    maxhr_bins: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str], np.ndarray | None]:
    """Add bucketed features as categorical strings."""
    df = df.copy()
    cat_cols_model = list(cat_cols_model)

    if 'ST depression' in df.columns:
        df['STdep_bucket'] = pd.cut(
            df['ST depression'],
            bins=[-0.1, 0, 1, 2, 10],
            labels=['0', '0-1', '1-2', '2+'],
        ).astype(str)
        cat_cols_model.append('STdep_bucket')

    if 'Age' in df.columns:
        df['Age_bucket'] = pd.cut(
            df['Age'],
            bins=[0, 40, 50, 60, 70, 100],
            labels=['<40', '40-50', '50-60', '60-70', '70+'],
        ).astype(str)
        cat_cols_model.append('Age_bucket')

    if 'Cholesterol' in df.columns:
        df['Chol_bucket'] = pd.cut(
            df['Cholesterol'],
            bins=[0, 200, 240, 280, 1000],
            labels=['<200', '200-240', '240-280', '280+'],
        ).astype(str)
        cat_cols_model.append('Chol_bucket')

    if 'Max HR' in df.columns:
        if maxhr_bins is None:
            maxhr_bins = compute_quantile_bins(df['Max HR'], q=5)
        df['MaxHR_bucket'] = pd.cut(
            df['Max HR'],
            bins=maxhr_bins,
            labels=False,
            include_lowest=True,
        ).astype(str)
        df['MaxHR_bucket'] = 'HR_Q' + df['MaxHR_bucket']
        cat_cols_model.append('MaxHR_bucket')

    if 'BP' in df.columns:
        df['BP_bucket'] = pd.cut(
            df['BP'],
            bins=[0, 120, 140, 160, 300],
            labels=['<120', '120-140', '140-160', '160+'],
        ).astype(str)
        cat_cols_model.append('BP_bucket')

    return df, cat_cols_model, maxhr_bins


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add selective polynomial features."""
    df = df.copy()

    if 'Age' in df.columns:
        df['Age_squared'] = df['Age'] ** 2

    if 'Max HR' in df.columns:
        df['MaxHR_squared'] = df['Max HR'] ** 2

    if 'Max HR' in df.columns and 'BP' in df.columns:
        df['MaxHR_div_BP'] = df['Max HR'] / (df['BP'] + 1)

    return df


def _cat_idx(X: pd.DataFrame, cat_features: list[str]) -> list[int]:
    return [X.columns.get_loc(c) for c in cat_features if c in X.columns]


def run_catboost_cv_fast(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cat_features: list[str],
    n_splits: int = 3,
    random_state: int = 42,
    tag: str = "cv",
) -> tuple[float, float]:
    """Fast CV for hyperparameter search."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs = []

    cat_idx = _cat_idx(X, cat_features)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        try:
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), verbose=False)
        except CatBoostError:
            cpu_params = {k: v for k, v in params.items() if k not in ['task_type', 'devices']}
            cpu_params['task_type'] = 'CPU'
            model = CatBoostClassifier(**cpu_params)
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), verbose=False)

        va_pred = model.predict_proba(X_va)[:, 1]
        aucs.append(roc_auc_score(y_va, va_pred))

    return float(np.mean(aucs)), float(np.std(aucs))


def run_catboost_cv_full(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict,
    cat_features: list[str],
    n_splits: int = 5,
    random_state: int = 42,
    tag: str = "cv",
) -> pd.DataFrame:
    """Full CV with per-fold stats."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    cat_idx = _cat_idx(X, cat_features)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[{tag}] Fold {fold}/{n_splits}")
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(**params)
        try:
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True, verbose=100)
        except CatBoostError:
            cpu_params = {k: v for k, v in params.items() if k not in ['task_type', 'devices']}
            cpu_params['task_type'] = 'CPU'
            model = CatBoostClassifier(**cpu_params)
            model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True, verbose=100)

        va_pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, va_pred)
        best_it = model.get_best_iteration() or params.get('iterations', 1000)
        rows.append({'fold': fold, 'roc_auc': auc, 'best_iteration': int(best_it)})
        print(f"[{tag}] Fold {fold} AUC: {auc:.6f}, Best iteration: {best_it}")

    return pd.DataFrame(rows)


def train_and_predict_seeds(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    params: dict,
    cat_features: list[str],
    n_splits: int = 5,
    seeds: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train with multiple seeds and average predictions."""
    if seeds is None:
        seeds = [0, 1, 2]

    test_preds_all = []
    oof_preds_all = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Training with random_seed={seed}")
        print(f"{'=' * 60}")

        params_seed = params.copy()
        params_seed['random_seed'] = seed

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        test_pred = np.zeros(len(X_test))
        oof_pred = np.zeros(len(X_train))

        cat_idx = _cat_idx(X_train, cat_features)

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = CatBoostClassifier(**params_seed)
            try:
                model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True, verbose=False)
            except CatBoostError:
                cpu_params = {k: v for k, v in params_seed.items() if k not in ['task_type', 'devices']}
                cpu_params['task_type'] = 'CPU'
                model = CatBoostClassifier(**cpu_params)
                model.fit(X_tr, y_tr, cat_features=cat_idx, eval_set=(X_va, y_va), use_best_model=True, verbose=False)

            va_pred = model.predict_proba(X_va)[:, 1]
            oof_pred[va_idx] = va_pred
            test_pred += model.predict_proba(X_test)[:, 1] / n_splits

            fold_auc = roc_auc_score(y_va, va_pred)
            print(f"  Fold {fold} AUC: {fold_auc:.6f}")

        oof_auc = roc_auc_score(y_train, oof_pred)
        print(f"Seed {seed} OOF AUC: {oof_auc:.6f}")

        test_preds_all.append(test_pred)
        oof_preds_all.append(oof_pred)

    test_pred_final = np.mean(test_preds_all, axis=0)
    oof_pred_final = np.mean(oof_preds_all, axis=0)
    final_oof_auc = roc_auc_score(y_train, oof_pred_final)
    print(f"\n{'=' * 60}")
    print(f"FINAL OOF AUC (averaged across {len(seeds)} seeds): {final_oof_auc:.6f}")
    print(f"{'=' * 60}")

    return test_pred_final, oof_pred_final, float(final_oof_auc)


def get_search_configs() -> list[dict]:
    """Define hyperparameter search space."""
    base_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': 2000,
        'od_type': 'Iter',
        'od_wait': 100,
        'learning_rate': 0.08,
        'task_type': 'CPU',
        'verbose': False,
    }

    configs = []
    configs.append({'name': 'baseline_d5', 'params': {**base_params, 'depth': 5}})
    configs.append({'name': 'deeper_d6', 'params': {**base_params, 'depth': 6}})
    configs.append({'name': 'deeper_d7', 'params': {**base_params, 'depth': 7}})
    configs.append({'name': 'd6_l2reg5', 'params': {**base_params, 'depth': 6, 'l2_leaf_reg': 5}})
    configs.append({'name': 'd6_l2reg10', 'params': {**base_params, 'depth': 6, 'l2_leaf_reg': 10}})
    configs.append({'name': 'd6_rs1', 'params': {**base_params, 'depth': 6, 'random_strength': 1.0}})
    configs.append({'name': 'd6_bt1', 'params': {**base_params, 'depth': 6, 'bagging_temperature': 1.0}})
    configs.append({'name': 'd6_mdl10', 'params': {**base_params, 'depth': 6, 'min_data_in_leaf': 10}})
    configs.append({
        'name': 'd6_bayesian',
        'params': {**base_params, 'depth': 6, 'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.0},
    })
    configs.append({
        'name': 'd6_bernoulli',
        'params': {**base_params, 'depth': 6, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8},
    })
    configs.append({
        'name': 'd7_l2reg5_rs1',
        'params': {**base_params, 'depth': 7, 'l2_leaf_reg': 5, 'random_strength': 1.0},
    })
    configs.append({
        'name': 'd6_auto_weights',
        'params': {**base_params, 'depth': 6, 'auto_class_weights': 'Balanced'},
    })

    return configs


if __name__ == "__main__":
    print("Heart Disease Improvement Utilities - Ready")
