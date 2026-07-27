"""
Image preprocessing and robustness transformations.
"""

from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str, rgb: bool = True) -> Image.Image:
    img = Image.open(path)
    if rgb:
        img = img.convert("RGB")
    return img


def resize_image(img: Image.Image, size: int = 224) -> Image.Image:
    return ImageOps.fit(img, (size, size), Image.BILINEAR)


def center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    return ImageOps.fit(img, (size, size), Image.BICUBIC, centering=(0.5, 0.5))


def jpeg_compress(img: Image.Image, quality: int = 70) -> Image.Image:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def add_gaussian_noise(img: Image.Image, sigma: float = 5.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def random_crop(img: Image.Image, crop_size: int = 200) -> Image.Image:
    width, height = img.size
    if crop_size >= min(width, height):
        return img
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))


def apply_transformations(img: Image.Image) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """
    Returns JPEG-compressed, resized, cropped, and noisy variants.
    """
    jpeg_img = jpeg_compress(img, quality=70)
    resized_img = resize_image(img, size=180)
    cropped_img = random_crop(img, crop_size=180)
    noisy_img = add_gaussian_noise(img, sigma=8.0)
    return jpeg_img, resized_img, cropped_img, noisy_img
