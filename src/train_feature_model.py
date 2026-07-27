"""
Train forensic feature-based classifiers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.forensic_features import build_feature_matrix, collect_image_paths


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def train_feature_models(data_dir: str = "data", outputs_dir: str = "outputs") -> Dict[str, Path]:
    paths, labels = collect_image_paths(data_dir)
    feature_result = build_feature_matrix(paths)

    X = feature_result.features
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "svm": Pipeline(
            [("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=42))]
        ),
        "log_reg": Pipeline(
            [("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=2000, random_state=42))]
        ),
    }

    outputs = Path(outputs_dir) / "models"
    outputs.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(outputs_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = []
    saved_paths: Dict[str, Path] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        metrics = _metrics(y_val, probs)
        results.append({"model": name, **metrics})
        out_path = outputs / f"{name}_forensic.pkl"
        joblib.dump(
            {"model": model, "feature_names": feature_result.names},
            out_path,
        )
        saved_paths[name] = out_path

    results_df = pd.DataFrame(results)
    results_df.to_csv(plots_dir / "forensic_model_results.csv", index=False)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train forensic feature models")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_feature_models(data_dir=args.data_dir, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
