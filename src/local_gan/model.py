"""Explicit CPU DCGAN baseline. No downloads or hidden training."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim.adam import Adam

from .core import DataError, image_array, image_paths, validate_settings


def networks(latent: int = 100, width: int = 128) -> tuple[Any, Any]:
    """Build the notebook's 64x64 RGB generator and discriminator topology."""
    generator: list[Any] = [
        nn.ConvTranspose2d(latent, width * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(width * 8),
        nn.ReLU(True),
    ]
    for left, right in [(8, 4), (4, 2), (2, 1)]:
        generator += [
            nn.ConvTranspose2d(width * left, width * right, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * right),
            nn.ReLU(True),
        ]
    generator += [nn.ConvTranspose2d(width, 3, 4, 2, 1, bias=False), nn.Tanh()]
    discriminator: list[Any] = [
        nn.Conv2d(3, width, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    for left, right in [(1, 2), (2, 4), (4, 8)]:
        discriminator += [
            nn.Conv2d(width * left, width * right, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * right),
            nn.LeakyReLU(0.2, inplace=True),
        ]
    discriminator += [
        nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid(),
        nn.Flatten(0),
    ]
    g, d = nn.Sequential(*generator), nn.Sequential(*discriminator)

    def initialize(module: Any) -> None:
        if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
            nn.init.normal_(module.weight, 0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1, 0.02)
            nn.init.zeros_(module.bias)

    g.apply(initialize)
    d.apply(initialize)
    return g, d


def generate(generator: Any, noise: Any) -> Any:
    """Sample without gradients or changing BatchNorm buffers; restore mode."""
    was_training = generator.training
    generator.eval()
    try:
        with torch.no_grad():
            return generator(noise).cpu()
    finally:
        generator.train(was_training)


def train(
    folder: Path,
    output: Path,
    epochs: int = 30,
    batch_size: int = 128,
    latent: int = 100,
    width: int = 128,
    learning_rate: float = 0.0004,
    max_steps: int = 100,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train both adversaries on local images, capped by total optimizer steps."""
    validate_settings(epochs, batch_size, latent, width, learning_rate, max_steps)
    paths = image_paths(folder)
    if len(paths) < 2:
        raise DataError("At least two images are required for training.")
    if output.exists():
        raise DataError("Output already exists. Choose a new output folder.")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    g, d = networks(latent, width)
    og = Adam(g.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    od = Adam(d.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    steps = 0
    for _ in range(epochs):
        order = torch.randperm(len(paths)).tolist()
        for start in range(0, len(paths), batch_size):
            selected = order[start : start + batch_size]
            if len(selected) < 2:
                continue  # Explicitly drop a singleton tail.
            real = torch.from_numpy(np.stack([image_array(paths[i]) for i in selected]))
            od.zero_grad(set_to_none=True)
            fake = g(torch.randn(len(selected), latent, 1, 1))
            dl = loss(d(real), torch.ones(len(selected))) + loss(
                d(fake.detach()), torch.zeros(len(selected))
            )
            dl.backward()
            od.step()
            og.zero_grad(set_to_none=True)
            for parameter in d.parameters():
                parameter.requires_grad_(False)
            gl = loss(d(fake), torch.ones(len(selected)))
            if not bool(torch.isfinite(gl)) or not bool(torch.isfinite(dl)):
                raise DataError("Nonfinite training loss; no checkpoint saved.")
            gl.backward()
            og.step()
            for parameter in d.parameters():
                parameter.requires_grad_(True)
            steps += 1
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
    output.mkdir(parents=True)
    torch.save(
        {
            "format": 1,
            "latent": latent,
            "width": width,
            "generator": g.state_dict(),
            "discriminator": d.state_dict(),
            "seed": seed,
            "steps": steps,
        },
        output / "checkpoint.pt",
    )
    provenance = {
        "format": 1,
        "input_sha256": sources,
        "checkpoint_sha256": hashlib.sha256(
            (output / "checkpoint.pt").read_bytes()
        ).hexdigest(),
        "settings": {
            "epochs": epochs,
            "batch_size": batch_size,
            "latent": latent,
            "width": width,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "seed": seed,
        },
        "steps": steps,
        "quality_claim": (
            "None. Training completion and hashes do not establish face quality."
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return {
        "steps": steps,
        "generator_loss": float(gl.detach()),
        "discriminator_loss": float(dl.detach()),
    }


def sample(
    checkpoint: Path, output: Path, count: int = 4, seed: int = 42
) -> list[Path]:
    """Load a locally trusted tensor checkpoint and save deterministic PNGs."""
    if count < 1 or count > 64:
        raise DataError("count must be between 1 and 64.")
    if output.exists():
        raise DataError("Output already exists. Choose a new output folder.")
    if not checkpoint.is_file():
        raise DataError("Checkpoint not found. Train a model first.")
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if data["format"] != 1:
            raise ValueError("format")
        latent, width = int(data["latent"]), int(data["width"])
        if not 1 <= latent <= 1024 or not 1 <= width <= 512:
            raise ValueError("dimensions")
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        g, _ = networks(latent, width)
        g.load_state_dict(data["generator"])
        images = generate(g, torch.randn(count, latent, 1, 1))
        if not bool(torch.isfinite(images).all()):
            raise ValueError("nonfinite generated pixels")
    except (
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise DataError(
            "Invalid checkpoint. Use a trusted local-dcgan checkpoint."
        ) from exc
    output.mkdir(parents=True)
    paths = []
    for index, image in enumerate(images):
        pixels = (
            ((image.clamp(-1, 1).numpy().transpose(1, 2, 0) + 1) * 127.5)
            .round()
            .astype("uint8")
        )
        path = output / f"sample-{index:03d}.png"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "format": 1,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "seed": seed,
                "count": count,
                "samples": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "quality_claim": (
                    "None. Samples are linked to this checkpoint, "
                    "not quality certified."
                ),
            },
            indent=2,
        )
    )
    return paths
