"""Thin local-only command interface."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from .core import DataError, image_paths

app = typer.Typer(
    help="Local 64x64 DCGAN baseline. No downloads or pretrained weights."
)
Option = typer.Option


@app.callback()
def configure() -> None:
    """Set up optional dotenv configuration and logs."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)


@app.command()
def check_data(folder: Annotated[Path, Option(envvar="GAN_IMAGES")]) -> None:
    """Decode every top-level PNG/JPEG and report the count."""
    try:
        typer.echo(
            json.dumps({"images": len(image_paths(folder)), "shape": [3, 64, 64]})
        )
    except DataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command()
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
) -> None:
    """Train from scratch on CPU with an explicit total step cap."""
    try:
        from .model import train as run

        typer.echo(
            json.dumps(
                run(
                    folder,
                    output,
                    epochs,
                    batch_size,
                    latent,
                    width,
                    learning_rate,
                    max_steps,
                    seed,
                )
            )
        )
    except (DataError, OSError, ImportError) as exc:
        typer.echo(f"{exc} Install the ml extra for training.", err=True)
        raise typer.Exit(2) from exc


@app.command()
def sample(checkpoint: Path, output: Path, count: int = 4, seed: int = 42) -> None:
    """Save samples from a trusted local checkpoint."""
    try:
        from .model import sample as run

        typer.echo(
            json.dumps(
                {"images": [str(p) for p in run(checkpoint, output, count, seed)]}
            )
        )
    except (DataError, OSError, ImportError) as exc:
        typer.echo(f"{exc} Install the ml extra for sampling.", err=True)
        raise typer.Exit(2) from exc
