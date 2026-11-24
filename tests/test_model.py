from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
import torch
from typer.testing import CliRunner

from local_gan.cli import app
from local_gan.core import DataError
from local_gan.model import generate, networks, sample, train


def fixture_images(path: Path, count: int = 3) -> Path:
    path.mkdir()
    rng = np.random.default_rng(11)
    for i in range(count):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype="uint8")).save(
            path / f"{i}.png"
        )
    return path


def test_network_and_eval_buffers() -> None:
    torch.set_num_threads(1)
    g, d = networks(3, 2)
    noise = torch.randn(2, 3, 1, 1)
    before = {k: v.clone() for k, v in g.state_dict().items()}
    output = generate(g, noise)
    assert g.training and output.shape == (2, 3, 64, 64)
    assert output.min() >= -1 and output.max() <= 1
    assert d(output).shape == (2,)
    assert all(torch.equal(before[k], v) for k, v in g.state_dict().items())
    g.eval()
    generate(g, noise)
    assert not g.training


def test_real_training_checkpoint_sampling(tmp_path: Path) -> None:
    images = fixture_images(tmp_path / "images")
    torch.manual_seed(42)
    initial_g, initial_d = networks(3, 2)
    report = train(
        images,
        tmp_path / "model",
        epochs=4,
        batch_size=2,
        latent=3,
        width=2,
        max_steps=2,
    )
    assert (
        report["steps"] == 2
        and np.isfinite(report["generator_loss"])
        and np.isfinite(report["discriminator_loss"])
    )
    data = torch.load(tmp_path / "model/checkpoint.pt", weights_only=True)
    for key, initial in [("generator", initial_g), ("discriminator", initial_d)]:
        assert any(
            not torch.equal(v, data[key][k])
            for k, v in initial.state_dict().items()
            if "weight" in k
        )
    p = sample(tmp_path / "model/checkpoint.pt", tmp_path / "samples", 2)
    q = sample(tmp_path / "model/checkpoint.pt", tmp_path / "again", 2)
    assert len(p) == 2 and p[0].read_bytes() == q[0].read_bytes()
    with Image.open(p[0]) as image:
        assert image.size == (64, 64) and image.mode == "RGB"
    with pytest.raises(DataError):
        sample(tmp_path / "model/checkpoint.pt", tmp_path / "samples")
    with pytest.raises(DataError):
        train(images, tmp_path / "model", latent=3, width=2)


def test_singleton_and_bad_checkpoint(tmp_path: Path) -> None:
    images = fixture_images(tmp_path / "one", 1)
    with pytest.raises(DataError):
        train(images, tmp_path / "model")
    for count in [0, 65]:
        with pytest.raises(DataError):
            sample(tmp_path / "none", tmp_path / "out", count)
    torch.save({"format": 2}, tmp_path / "bad.pt")
    with pytest.raises(DataError):
        sample(tmp_path / "bad.pt", tmp_path / "out")
    torch.save({"format": 1, "latent": 0, "width": 2}, tmp_path / "bad.pt")
    with pytest.raises(DataError):
        sample(tmp_path / "bad.pt", tmp_path / "out")


def test_cli_real_train_sample(tmp_path: Path) -> None:
    images = fixture_images(tmp_path / "images", 2)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            str(images),
            str(tmp_path / "model"),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--latent",
            "3",
            "--width",
            "2",
            "--max-steps",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "sample",
            str(tmp_path / "model/checkpoint.pt"),
            str(tmp_path / "out"),
            "--count",
            "1",
        ],
    )
    assert result.exit_code == 0 and (tmp_path / "out/sample-000.png").exists()


def test_corrupt_checkpoint(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.pt"
    p.write_bytes(b"not a torch checkpoint")
    with pytest.raises(DataError):
        sample(p, tmp_path / "out")
