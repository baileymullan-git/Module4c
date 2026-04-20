"""
Neural-network pipeline for breast cancer recurrence prediction.

Colab usage:
    !pip -q install xlrd==2.0.1
    !python mullan_module4c.py --data_path breast-cancer.xls
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TARGET_COL = "Class"
POSITIVE_CLASS = "recurrence-events"

# Allowed category labels in the original UCI breast-cancer dataset.
KNOWN_RANGES: Dict[str, set[str]] = {
    "tumor-size": {
        "0-4",
        "5-9",
        "10-14",
        "15-19",
        "20-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
    },
    "inv-nodes": {"0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26"},
}


@dataclass
class ModelConfig:
    hidden_units: Tuple[int, ...]
    alpha: float
    learning_rate_init: float
    max_iter: int


def set_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def normalize_excel_range(value: object, column_name: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        dt = pd.to_datetime(value)
        candidates = [f"{dt.day}-{dt.month}", f"{dt.month}-{dt.year % 100}"]
        allowed = KNOWN_RANGES.get(column_name, set())
        for candidate in candidates:
            if candidate in allowed:
                return candidate
        # Fallback for unforeseen values.
        return candidates[0]
    return str(value).strip()


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    df = pd.read_excel(data_path, engine="xlrd")
    if TARGET_COL not in df.columns:
        # Some spreadsheets place target as final unlabeled column.
        df.columns = [*df.columns[:-1], TARGET_COL]

    for col in ("tumor-size", "inv-nodes"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_excel_range(x, col))

    # Replace unknown markers with explicit category so we don't drop rows.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"?": "Unknown"})

    return df


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
    )


def build_model(config: ModelConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_units,
        activation="relu",
        solver="adam",
        alpha=config.alpha,
        learning_rate_init=config.learning_rate_init,
        batch_size=16,
        max_iter=config.max_iter,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )


def best_threshold_for_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def evaluate_config_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    categorical_cols: List[str],
    numeric_cols: List[str],
    config: ModelConfig,
    n_splits: int = 5,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    auc_scores: List[float] = []
    ap_scores: List[float] = []
    f1_scores: List[float] = []
    thresholds: List[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        preprocessor = build_preprocessor(categorical_cols, numeric_cols)
        X_train_p = preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_p = preprocessor.transform(X_val).astype(np.float32)

        model = build_model(config=config)
        model.fit(X_train_p, y_train)
        y_prob = model.predict_proba(X_val_p)[:, 1]
        threshold = best_threshold_for_f1(y_val, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_prob))
        ap_scores.append(average_precision_score(y_val, y_prob))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        thresholds.append(threshold)

    return {
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores)),
        "ap_mean": float(np.mean(ap_scores)),
        "f1_mean": float(np.mean(f1_scores)),
        "best_threshold": float(np.mean(thresholds)),
    }


def run_training(data_path: str) -> None:
    set_seed()
    df = load_and_clean_data(data_path)

    y = (df[TARGET_COL] == POSITIVE_CLASS).astype(int).values
    X = df.drop(columns=[TARGET_COL]).copy()

    numeric_cols = [c for c in X.columns if c == "deg-malig"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    configs = [
        ModelConfig((64, 32), 1e-4, 1e-3, 1200),
        ModelConfig((128, 64), 5e-4, 8e-4, 1400),
        ModelConfig((128, 64, 32), 1e-4, 8e-4, 1500),
        ModelConfig((96, 48), 1e-3, 5e-4, 1200),
        ModelConfig((64, 64, 32), 1e-5, 6e-4, 1300),
        ModelConfig((128, 32), 1e-4, 1e-3, 1200),
    ]

    print("Tuning configurations with 5-fold stratified CV...")
    cv_results = []
    for i, config in enumerate(configs, start=1):
        result = evaluate_config_cv(X_train, y_train, categorical_cols, numeric_cols, config, n_splits=5)
        cv_results.append((config, result))
        print(
            f"[{i}/{len(configs)}] units={config.hidden_units}, alpha={config.alpha}, "
            f"lr={config.learning_rate_init}, max_iter={config.max_iter} | "
            f"AUC={result['auc_mean']:.4f} (+/- {result['auc_std']:.4f}), "
            f"AP={result['ap_mean']:.4f}, F1={result['f1_mean']:.4f}"
        )

    # Prioritize ROC-AUC, then PR-AUC, then F1.
    best_config, best_result = max(
        cv_results,
        key=lambda x: (x[1]["auc_mean"], x[1]["ap_mean"], x[1]["f1_mean"]),
    )
    threshold = best_result["best_threshold"]

    print("\nBest configuration selected:")
    print(best_config)
    print(
        f"CV AUC={best_result['auc_mean']:.4f}, CV AP={best_result['ap_mean']:.4f}, "
        f"CV F1={best_result['f1_mean']:.4f}, threshold={threshold:.3f}"
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X_train_p = preprocessor.fit_transform(X_train).astype(np.float32)
    X_test_p = preprocessor.transform(X_test).astype(np.float32)

    model = build_model(config=best_config)
    model.fit(X_train_p, y_train)

    y_prob = model.predict_proba(X_test_p)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("\nHeld-out test performance:")
    print(f"ROC-AUC:    {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:     {average_precision_score(y_test, y_prob):.4f}")
    print(f"F1:         {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Precision:  {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:     {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Accuracy:   {accuracy_score(y_test, y_pred):.4f}")

    # Optional artifact exports for deployment/reuse.
    joblib.dump(
        {"preprocessor": preprocessor, "model": model, "threshold": threshold},
        "breast_cancer_recurrence_model.joblib",
    )
    pd.Series(y_prob, name="recurrence_probability").to_csv(
        "test_recurrence_probabilities.csv",
        index=False,
    )
    print("\nSaved model bundle: breast_cancer_recurrence_model.joblib")
    print("Saved test probabilities: test_recurrence_probabilities.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast cancer recurrence prediction with neural networks")
    parser.add_argument(
        "--data_path",
        type=str,
        default="breast-cancer.xls",
        help="Path to the breast cancer spreadsheet (.xls).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.data_path)
