"""
Forensic feature extraction for synthetic image detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


@dataclass
class FeatureResult:
    features: np.ndarray
    names: List[str]


def _fft_power_spectrum(gray: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    return power


def _spectral_entropy(power: np.ndarray, eps: float = 1e-12) -> float:
    p = power.flatten().astype(np.float64)
    p = p / (p.sum() + eps)
    entropy = -np.sum(p * np.log2(p + eps))
    return float(entropy)


def _high_freq_energy_ratio(power: np.ndarray) -> float:
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_radius = np.max(radius)
    high_mask = radius > (0.5 * max_radius)
    high_energy = power[high_mask].sum()
    total_energy = power.sum() + 1e-12
    return float(high_energy / total_energy)


def _noise_residual(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) - blur.astype(np.float32)


def _noise_correlation(noise: np.ndarray) -> float:
    shifted = np.roll(noise, shift=1, axis=1)
    num = np.mean((noise - noise.mean()) * (shifted - shifted.mean()))
    denom = np.std(noise) * np.std(shifted) + 1e-12
    return float(num / denom)


def _color_channel_correlation(img: np.ndarray) -> Tuple[float, float, float]:
    r, g, b = img[:, :, 0].flatten(), img[:, :, 1].flatten(), img[:, :, 2].flatten()
    corr_rg = np.corrcoef(r, g)[0, 1]
    corr_rb = np.corrcoef(r, b)[0, 1]
    corr_gb = np.corrcoef(g, b)[0, 1]
    return float(corr_rg), float(corr_rb), float(corr_gb)


def _texture_features(gray: np.ndarray) -> Dict[str, float]:
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 11), density=True)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0

    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    feats = {
        "edge_density": float(edge_density),
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
    }

    for i, v in enumerate(lbp_hist):
        feats[f"lbp_{i}"] = float(v)
    return feats


def _extract_features_from_rgb(img: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    power = _fft_power_spectrum(gray)
    spectral_entropy = _spectral_entropy(power)
    high_freq_ratio = _high_freq_energy_ratio(power)

    noise = _noise_residual(gray)
    noise_var = float(np.var(noise))
    noise_corr = _noise_correlation(noise)

    corr_rg, corr_rb, corr_gb = _color_channel_correlation(img)

    texture = _texture_features(gray)

    features = {
        "spectral_entropy": spectral_entropy,
        "high_freq_ratio": high_freq_ratio,
        "noise_variance": noise_var,
        "noise_correlation": noise_corr,
        "corr_rg": corr_rg,
        "corr_rb": corr_rb,
        "corr_gb": corr_gb,
    }
    features.update(texture)
    return features


def extract_features(image_path: str) -> Dict[str, float]:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _extract_features_from_rgb(img)


def extract_features_from_array(rgb_array: np.ndarray) -> Dict[str, float]:
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("Expected RGB array with shape (H, W, 3).")
    return _extract_features_from_rgb(rgb_array)


def build_feature_matrix(image_paths: List[str]) -> FeatureResult:
    records = []
    for path in image_paths:
        feats = extract_features(path)
        records.append(feats)

    names = sorted(records[0].keys())
    matrix = np.array([[r[n] for n in names] for r in records], dtype=np.float32)
    return FeatureResult(features=matrix, names=names)


def build_feature_matrix_from_arrays(images: List[np.ndarray]) -> FeatureResult:
    records = []
    for img in images:
        feats = extract_features_from_array(img)
        records.append(feats)

    names = sorted(records[0].keys())
    matrix = np.array([[r[n] for n in names] for r in records], dtype=np.float32)
    return FeatureResult(features=matrix, names=names)


def collect_image_paths(data_dir: str) -> Tuple[List[str], List[int]]:
    data_path = Path(data_dir)
    real_paths = sorted(
        [str(p) for p in (data_path / "real").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    synth_paths = sorted(
        [str(p) for p in (data_path / "synthetic").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not real_paths or not synth_paths:
        raise ValueError("Both data/real and data/synthetic must contain images.")

    paths = real_paths + synth_paths
    labels = [0] * len(real_paths) + [1] * len(synth_paths)
    return paths, labels
