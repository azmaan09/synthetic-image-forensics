"""
Faithfulness and interpretability analysis for CNN and forensic models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.cnn_detector import load_model
from src.forensic_features import build_feature_matrix, collect_image_paths


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[:, class_idx].sum()
        loss.backward()

        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def _overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cam_resized = np.uint8(255 * cam)
    heatmap = plt.cm.jet(cam_resized / 255.0)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return overlay


def run_gradcam(
    image_path: str, model_path: str, output_path: Path, img_size: int = 224
) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device=device)
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model, target_layer)

    preprocess = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    heatmap = cam.generate(input_tensor)
    img_np = np.array(img.resize((heatmap.shape[1], heatmap.shape[0])))
    overlay = _overlay_heatmap(img_np, heatmap)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def feature_importance_plot(model_path: str, data_dir: str, output_path: Path) -> Path:
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload["feature_names"]

    paths, _ = collect_image_paths(data_dir)
    feature_result = build_feature_matrix(paths)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-15:]
        names = [feature_names[i] for i in idx]
        values = importances[idx]
    else:
        values = np.mean(np.abs(model.coef_), axis=0)
        idx = np.argsort(values)[-15:]
        names = [feature_names[i] for i in idx]
        values = values[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh(names, values)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def shap_summary_plot(model_path: str, data_dir: str, output_path: Path) -> Optional[Path]:
    try:
        import shap
    except ImportError:
        return None

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_names = payload["feature_names"]
    paths, _ = collect_image_paths(data_dir)
    feature_result = build_feature_matrix(paths)
    X = feature_result.features

    sample = X[:200]
    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model, sample, feature_names=feature_names)
        shap_values = explainer(sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return output_path
    return None


def run_faithfulness_analysis(data_dir: str = "data", outputs_dir: str = "outputs") -> None:
    outputs_path = Path(outputs_dir) / "plots"
    outputs_path.mkdir(parents=True, exist_ok=True)

    cnn_path = Path(outputs_dir) / "models" / "cnn_resnet50.pt"
    forensic_path = Path(outputs_dir) / "models" / "random_forest_forensic.pkl"
    paths, labels = collect_image_paths(data_dir)

    if cnn_path.exists():
        run_gradcam(paths[0], str(cnn_path), outputs_path / "gradcam_example.png")

    if forensic_path.exists():
        feature_importance_plot(str(forensic_path), data_dir, outputs_path / "feature_importance.png")
        shap_summary_plot(str(forensic_path), data_dir, outputs_path / "shap_summary.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithfulness and interpretability analysis")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_faithfulness_analysis(data_dir=args.data_dir, outputs_dir=args.outputs_dir)


if __name__ == "__main__":
    main()
