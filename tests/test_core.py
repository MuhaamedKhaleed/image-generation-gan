from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from local_gan.core import DataError, image_array, image_paths, validate_settings


@pytest.fixture
def images(tmp_path: Path) -> Path:
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        Image.fromarray(np.full((70, 90, 3), i * 100, dtype="uint8")).save(
            folder / f"{i}.png"
        )
    return folder


def test_flat_folder_and_normalization(images: Path) -> None:
    paths = image_paths(images)
    assert len(paths) == 3
    a = image_array(paths[0])
    assert a.shape == (3, 64, 64) and a.dtype == np.float32
    assert a.max() == -1
    assert image_array(paths[-1]).min() == pytest.approx(200 / 127.5 - 1)


def test_grayscale_and_crop(tmp_path: Path) -> None:
    p = tmp_path / "gray.png"
    Image.new("L", (20, 100), 255).save(p)
    assert np.all(image_array(p) == 1)


@pytest.mark.parametrize("case", ["missing", "empty", "nested", "corrupt"])
def test_bad_folders(tmp_path: Path, case: str) -> None:
    folder = tmp_path / "input"
    if case != "missing":
        folder.mkdir()
    if case == "nested":
        (folder / "nested").mkdir()
        Image.new("RGB", (4, 4)).save(folder / "nested/a.png")
    if case == "corrupt":
        (folder / "a.png").write_text("broken")
    with pytest.raises(DataError):
        image_paths(folder)


@pytest.mark.parametrize(
    "args",
    [
        (0, 2, 2, 2, 0.001, 1),
        (1, 2, 1025, 2, 0.001, 1),
        (1, 2, 2, 513, 0.001, 1),
        (1, 1, 2, 2, 0.001, 1),
        (1, 2, 0, 2, 0.001, 1),
        (1, 2, 2, 0, 0.001, 1),
        (1, 2, 2, 2, 0, 1),
        (1, 2, 2, 2, float("nan"), 1),
        (1, 2, 2, 2, 0.001, 0),
    ],
)
def test_settings(args: tuple[int, int, int, int, float, int]) -> None:
    with pytest.raises(DataError):
        validate_settings(*args)
