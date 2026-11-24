"""Generate synthetic colored-noise inputs, not a real image dataset."""

from pathlib import Path

import numpy as np
from PIL import Image

folder = Path("synthetic-demo/images")
folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
for i in range(3):
    Image.fromarray(rng.integers(0, 256, (72, 80, 3), dtype="uint8")).save(
        folder / f"{i}.png"
    )
