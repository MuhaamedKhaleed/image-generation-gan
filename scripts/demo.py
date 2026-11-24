"""Explicit synthetic wiring demo. No real-data accuracy or quality claim."""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "-1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ.setdefault(key, value)

    out = Path(sys.argv[1] if len(sys.argv) == 2 else "synthetic-run")
    if len(sys.argv) > 2:
        raise SystemExit("Usage: python scripts/demo.py [NEW_OUTPUT_DIRECTORY]")
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from PIL import Image

    from local_gan.model import sample, train

    torch.set_num_threads(1)
    folder = out / "images"
    folder.mkdir()
    rng = np.random.default_rng(42)
    for i in range(3):
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype="uint8")).save(
            folder / f"{i}.png"
        )
    scores = train(
        folder, out / "model", epochs=2, batch_size=2, latent=3, width=2, max_steps=2
    )
    images = sample(out / "model/checkpoint.pt", out / "samples", 2)
    assert len(images) == 2 and scores["steps"] == 2
    result = {"training": scores, "sample_count": len(images)}
    scope = "invented_noise_tiny_DCGAN_not_realistic_generation_quality"
    report = {
        "synthetic_only": True,
        "scope": scope,
        "quality_claim": False,
        "result": result,
    }
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"synthetic_only": True, "report": str(out / "report.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
