"""
Data loading utilities and optional dataset download helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import random
import shutil
import tarfile
import zipfile

import requests
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms


@dataclass
class DataLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    class_names: List[str]


def get_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoaders:
    dataset_root = Path(data_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Data directory not found: {dataset_root}")

    dataset = ImageFolder(root=str(dataset_root), transform=get_transforms(img_size=img_size, train=True))
    if len(dataset) == 0:
        raise ValueError("No images found. Ensure data/real and data/synthetic contain images.")

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_ds.dataset.transform = get_transforms(img_size=img_size, train=True)
    val_ds.dataset.transform = get_transforms(img_size=img_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataLoaders(train_loader=train_loader, val_loader=val_loader, class_names=dataset.classes)


def download_and_extract(url: str, dest_dir: str) -> Path:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    archive_path = dest_path / filename

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {filename}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_path)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_path)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    return dest_path


def _prepare_dest(dest_dir: Optional[str], default_subdir: str) -> str:
    return dest_dir if dest_dir else str(Path("data") / default_subdir)


def download_stable_diffusion_samples(url: Optional[str], dest_dir: Optional[str] = None) -> Path:
    """
    Download a Stable Diffusion dataset archive.
    Provide a direct URL to a zip/tar archive.
    """
    if not url:
        raise ValueError("Stable Diffusion URL is required.")
    dest = _prepare_dest(dest_dir, "synthetic")
    return download_and_extract(url, dest)


def download_stylegan_samples(url: Optional[str], dest_dir: Optional[str] = None) -> Path:
    """
    Download a StyleGAN dataset archive.
    Provide a direct URL to a zip/tar archive.
    """
    if not url:
        raise ValueError("StyleGAN URL is required.")
    dest = _prepare_dest(dest_dir, "synthetic")
    return download_and_extract(url, dest)


def download_imagenet_subset(url: Optional[str], dest_dir: Optional[str] = None) -> Path:
    """
    Download an ImageNet subset archive (user-provided URL).
    """
    if not url:
        raise ValueError("ImageNet subset URL is required.")
    dest = _prepare_dest(dest_dir, "real")
    return download_and_extract(url, dest)


def download_coco_subset(url: Optional[str], dest_dir: Optional[str] = None) -> Path:
    """
    Download a COCO subset archive (user-provided URL).
    """
    if not url:
        raise ValueError("COCO subset URL is required.")
    dest = _prepare_dest(dest_dir, "real")
    return download_and_extract(url, dest)


def copy_subset(src_dir: str, dest_dir: str, max_images: int = 1000, seed: int = 42) -> None:
    """
    Copy a random subset of images to dest_dir for quick experiments.
    """
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    images = [p for p in src_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if not images:
        raise ValueError(f"No images found in {src_dir}")

    random.Random(seed).shuffle(images)
    for img in images[:max_images]:
        target = dest_path / img.name
        shutil.copy2(img, target)
