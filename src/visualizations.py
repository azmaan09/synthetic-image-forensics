"""
Generate a comprehensive set of project visuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from torch.utils.data import DataLoader, Dataset

from src.faithfulness_analysis import run_gradcam, feature_importance_plot, shap_summary_plot
from src.forensic_features import build_feature_matrix, collect_image_paths, _fft_power_spectrum
from src.preprocessing import apply_transformations
from src.cnn_detector import load_model
from src.data_loader import get_transforms


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


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _build_labels(paths: List[str]) -> List[int]:
    labels = []
    for p in paths:
        if "/real/" in p.replace("\\", "/"):
            labels.append(0)
        else:
            labels.append(1)
    return labels


def plot_frequency_examples(paths: List[str], output_path: Path) -> None:
    img = Image.open(paths[0]).convert("L")
    power = _fft_power_spectrum(np.array(img))
    plt.figure(figsize=(4, 4))
    plt.imshow(np.log1p(power), cmap="magma")
    plt.title("FFT Power Spectrum (Example)")
    plt.axis("off")
    _save_figure(output_path)


def plot_class_distribution(labels: List[int], output_path: Path) -> None:
    plt.figure(figsize=(4, 3))
    sns.countplot(x=labels)
    plt.title("Class Distribution")
    plt.xlabel("Label (0=real, 1=synthetic)")
    plt.ylabel("Count")
    _save_figure(output_path)


def plot_sample_grid(paths: List[str], output_path: Path, title: str) -> None:
    sample = paths[:9]
    plt.figure(figsize=(6, 6))
    for i, path in enumerate(sample):
        img = Image.open(path).convert("RGB")
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(title)
    _save_figure(output_path)


def plot_feature_distributions(data_dir: str, output_dir: Path) -> None:
    paths, labels = collect_image_paths(data_dir)
    feature_result = build_feature_matrix(paths)
    df = pd.DataFrame(feature_result.features, columns=feature_result.names)
    df["label"] = labels

    for feature in ["high_freq_ratio", "spectral_entropy", "noise_variance", "noise_correlation"]:
        if feature in df.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df, x="label", y=feature)
            plt.title(f"{feature} by class")
            _save_figure(output_dir / f"feature_{feature}.png")

    plt.figure(figsize=(8, 6))
    corr = df.drop(columns=["label"]).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    _save_figure(output_dir / "feature_correlation_matrix.png")

    lbp_cols = [c for c in df.columns if c.startswith("lbp_")]
    if lbp_cols:
        plt.figure(figsize=(6, 4))
        plt.plot(df[lbp_cols].mean().values)
        plt.title("Mean LBP Histogram")
        plt.xlabel("LBP Bin")
        plt.ylabel("Frequency")
        _save_figure(output_dir / "lbp_histogram.png")


def plot_transformations_example(data_dir: str, output_path: Path) -> None:
    paths, _ = collect_image_paths(data_dir)
    img = Image.open(paths[0]).convert("RGB")
    jpeg_img, resized_img, cropped_img, noisy_img = apply_transformations(img)

    plt.figure(figsize=(8, 6))
    for i, (title, im) in enumerate(
        [("original", img), ("jpeg", jpeg_img), ("resize", resized_img), ("crop", cropped_img), ("noise", noisy_img)]
    ):
        plt.subplot(2, 3, i + 1)
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    _save_figure(output_path)


def plot_noise_and_gradients(paths: List[str], output_dir: Path) -> None:
    img = Image.open(paths[0]).convert("L")
    arr = np.array(img).astype(np.float32)
    blur = np.clip(arr - np.clip(arr, 0, 255), 0, 255)
    plt.figure(figsize=(4, 4))
    plt.imshow(blur, cmap="gray")
    plt.title("Noise Residual (Example)")
    plt.axis("off")
    _save_figure(output_dir / "noise_residual_example.png")

    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    plt.figure(figsize=(4, 4))
    plt.imshow(grad, cmap="inferno")
    plt.title("Gradient Magnitude (Example)")
    plt.axis("off")
    _save_figure(output_dir / "gradient_magnitude_example.png")


def _cnn_probs(paths: List[str], labels: List[int], model_path: Path) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(str(model_path), device=device)
    transform = get_transforms(img_size=224, train=False)
    dataset = ImagePathDataset(paths, labels, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


def _forensic_probs(paths: List[str], model_path: Path) -> np.ndarray:
    import joblib

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_result = build_feature_matrix(paths)
    return model.predict_proba(feature_result.features)[:, 1]


def plot_roc_pr_curves(labels: List[int], probs: np.ndarray, output_prefix: Path, title_prefix: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"{title_prefix} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    _save_figure(output_prefix.with_name(f"{output_prefix.stem}_roc.png"))

    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.title(f"{title_prefix} Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    _save_figure(output_prefix.with_name(f"{output_prefix.stem}_pr.png"))


def plot_confusion(labels: List[int], probs: np.ndarray, output_path: Path, title: str) -> None:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _save_figure(output_path)


def generate_all_visuals(data_dir: str = "data", outputs_dir: str = "outputs") -> None:
    outputs_path = Path(outputs_dir) / "plots"
    outputs_path.mkdir(parents=True, exist_ok=True)

    paths, _ = collect_image_paths(data_dir)
    labels = _build_labels(paths)

    plot_frequency_examples(paths, outputs_path / "fft_example.png")
    plot_feature_distributions(data_dir, outputs_path)
    plot_transformations_example(data_dir, outputs_path / "transformations_example.png")
    plot_noise_and_gradients(paths, outputs_path)
    plot_class_distribution(labels, outputs_path / "class_distribution.png")

    real_paths = [p for p in paths if "/real/" in p.replace("\\", "/")]
    syn_paths = [p for p in paths if "/synthetic/" in p.replace("\\", "/")]
    if real_paths:
        plot_sample_grid(real_paths, outputs_path / "real_samples_grid.png", "Real Samples")
    if syn_paths:
        plot_sample_grid(syn_paths, outputs_path / "synthetic_samples_grid.png", "Synthetic Samples")

    cnn_path = Path(outputs_dir) / "models" / "cnn_resnet50.pt"
    forensic_path = Path(outputs_dir) / "models" / "random_forest_forensic.pkl"

    if cnn_path.exists():
        run_gradcam(paths[0], str(cnn_path), outputs_path / "gradcam_example.png")
        cnn_probs = _cnn_probs(paths, labels, cnn_path)
        plot_roc_pr_curves(labels, cnn_probs, outputs_path / "cnn_metrics.png", "CNN")
        plot_confusion(labels, cnn_probs, outputs_path / "cnn_confusion.png", "CNN Confusion Matrix")

    if forensic_path.exists():
        feature_importance_plot(str(forensic_path), data_dir, outputs_path / "feature_importance.png")
        shap_summary_plot(str(forensic_path), data_dir, outputs_path / "shap_summary.png")
        forensic_probs = _forensic_probs(paths, forensic_path)
        plot_roc_pr_curves(labels, forensic_probs, outputs_path / "forensic_metrics.png", "Forensic")
        plot_confusion(labels, forensic_probs, outputs_path / "forensic_confusion.png", "Forensic Confusion Matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project visualizations")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_visuals(data_dir=args.data_dir, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
