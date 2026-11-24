"""Strict flat-folder image inputs and reproducible preprocessing."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps


class DataError(ValueError):
    """An image input or training setting is not usable."""


def image_paths(folder: Path) -> list[Path]:
    """Return supported top-level files, rejecting empty or unreadable inputs."""
    if not folder.is_dir():
        raise DataError(f"Image folder not found: {folder}. Supply an existing folder.")
    paths = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise DataError("No PNG/JPEG images found directly in the folder.")
    for p in paths:
        image_array(p)
    return paths


def image_array(path: Path) -> NDArray[np.float32]:
    """Center-crop RGB images to 64 pixels and scale CHW values to [-1, 1]."""
    try:
        with Image.open(path) as image:
            rgb = ImageOps.fit(
                image.convert("RGB"), (64, 64), method=Image.Resampling.BILINEAR
            )
            return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / np.float32(
                127.5
            ) - np.float32(1)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DataError(f"Cannot read image {path}. Supply a valid PNG/JPEG.") from exc


def validate_settings(
    epochs: int,
    batch_size: int,
    latent: int,
    width: int,
    learning_rate: float,
    max_steps: int,
) -> None:
    """Fail early for unsupported settings without allocating a network."""
    import math

    if min(epochs, latent, width, max_steps) < 1 or batch_size < 2:
        raise DataError(
            "epochs, latent, width, max-steps must be positive; "
            "batch-size must be at least 2."
        )
    if latent > 1024 or width > 512:
        raise DataError("latent must be at most 1024 and width at most 512.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise DataError("learning-rate must be finite and positive.")
