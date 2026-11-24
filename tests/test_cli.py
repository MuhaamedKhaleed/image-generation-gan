from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from local_gan.cli import app

runner = CliRunner()


def test_checks_and_help(tmp_path: Path, monkeypatch: object) -> None:
    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    result = runner.invoke(app, ["check-data", "--folder", str(tmp_path)])
    assert result.exit_code == 0 and '"images": 1' in result.stdout
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["check-data", "--folder", str(tmp_path / "missing")])
    assert result.exit_code == 2 and "folder not found" in result.output


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") is None,
    reason="ML extra not installed",
)
def test_training_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["train", str(tmp_path), str(tmp_path / "out"), "--epochs", "0"]
    )
    assert result.exit_code == 2 and "must be positive" in result.output


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("torch") is None,
    reason="ML extra not installed",
)
def test_sampling_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["sample", str(tmp_path / "no.pt"), str(tmp_path / "out")]
    )
    assert result.exit_code == 2 and "Checkpoint not found" in result.output
