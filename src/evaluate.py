"""
Model evaluation and robustness testing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.cnn_detector import load_model
from src.data_loader import get_transforms
from src.forensic_features import build_feature_matrix, build_feature_matrix_from_arrays, collect_image_paths
from src.preprocessing import apply_transformations


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def evaluate_cnn(paths: List[str], labels: List[int], model_path: str, batch_size: int = 32) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)
    model.eval()

    transform = get_transforms(img_size=224, train=False)
    dataset = ImagePathDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, batch_labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(np.array(batch_labels))

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return _metrics(y_true, y_prob)


def evaluate_forensic(paths: List[str], labels: List[int], model_path: str) -> Dict[str, float]:
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_result = build_feature_matrix(paths)
    probs = model.predict_proba(feature_result.features)[:, 1]
    return _metrics(np.array(labels), probs)


def evaluate_hybrid(
    paths: List[str], labels: List[int], cnn_path: str, forensic_path: str
) -> Dict[str, float]:
    payload = joblib.load(forensic_path)
    feature_model = payload["model"]
    feature_result = build_feature_matrix(paths)
    feature_probs = feature_model.predict_proba(feature_result.features)[:, 1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = load_model(cnn_path, device=device)
    cnn.eval()
    transform = get_transforms(img_size=224, train=False)
    dataset = ImagePathDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    cnn_probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = cnn(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            cnn_probs.append(probs)

    cnn_probs = np.concatenate(cnn_probs)
    hybrid_probs = 0.5 * cnn_probs + 0.5 * feature_probs
    return _metrics(np.array(labels), hybrid_probs)


def evaluate_robustness(
    paths: List[str], labels: List[int], cnn_path: str, forensic_path: str
) -> pd.DataFrame:
    payload = joblib.load(forensic_path)
    feature_model = payload["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = load_model(cnn_path, device=device)
    cnn.eval()

    results = []
    for name in ["jpeg", "resize", "crop", "noise"]:
        transformed_images = []
        for path in paths:
            img = Image.open(path).convert("RGB")
            jpeg_img, resized_img, cropped_img, noisy_img = apply_transformations(img)
            transformed = {
                "jpeg": jpeg_img,
                "resize": resized_img,
                "crop": cropped_img,
                "noise": noisy_img,
            }[name]
            transformed_images.append(transformed)

        # CNN evaluation
        transform = get_transforms(img_size=224, train=False)
        tensors = torch.stack([transform(img) for img in transformed_images])
        cnn_probs = []
        with torch.no_grad():
            for i in range(0, len(tensors), 32):
                batch = tensors[i : i + 32].to(device)
                logits = cnn(batch)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                cnn_probs.append(probs)
        cnn_probs = np.concatenate(cnn_probs)
        cnn_metrics = _metrics(np.array(labels), cnn_probs)

        # Forensic evaluation on transformed images
        arrays = [np.array(img) for img in transformed_images]
        feature_matrix = build_feature_matrix_from_arrays(arrays)
        forensic_probs = feature_model.predict_proba(feature_matrix.features)[:, 1]
        forensic_metrics = _metrics(np.array(labels), forensic_probs)

        results.append(
            {
                "transformation": name,
                "cnn_accuracy": cnn_metrics["accuracy"],
                "cnn_precision": cnn_metrics["precision"],
                "cnn_recall": cnn_metrics["recall"],
                "cnn_auc": cnn_metrics["auc"],
                "forensic_accuracy": forensic_metrics["accuracy"],
                "forensic_precision": forensic_metrics["precision"],
                "forensic_recall": forensic_metrics["recall"],
                "forensic_auc": forensic_metrics["auc"],
            }
        )

    return pd.DataFrame(results)


def evaluate_all(data_dir: str = "data", outputs_dir: str = "outputs") -> Path:
    paths, labels = collect_image_paths(data_dir)
    outputs_path = Path(outputs_dir)
    outputs_path.mkdir(parents=True, exist_ok=True)
    plots_dir = outputs_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    cnn_path = outputs_path / "models" / "cnn_resnet50.pt"
    forensic_path = outputs_path / "models" / "random_forest_forensic.pkl"

    results = []
    if cnn_path.exists():
        results.append({"model": "cnn", **evaluate_cnn(paths, labels, str(cnn_path))})
    if forensic_path.exists():
        results.append({"model": "forensic", **evaluate_forensic(paths, labels, str(forensic_path))})
    if cnn_path.exists() and forensic_path.exists():
        results.append({"model": "hybrid", **evaluate_hybrid(paths, labels, str(cnn_path), str(forensic_path))})

    results_df = pd.DataFrame(results)
    results_df.to_csv(plots_dir / "model_comparison.csv", index=False)

    if cnn_path.exists() and forensic_path.exists():
        robust_df = evaluate_robustness(paths, labels, str(cnn_path), str(forensic_path))
        robust_df.to_csv(plots_dir / "robustness_results.csv", index=False)

    return plots_dir / "model_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models and compare performance")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_all(data_dir=args.data_dir, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
