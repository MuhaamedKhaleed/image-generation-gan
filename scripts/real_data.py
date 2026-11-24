"""Explicit real-data CPU development recipe. Never runs on import."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Prepared local source directory")
    parser.add_argument(
        "--base-model", help="Trusted compatible local DistilBERT directory"
    )
    parser.add_argument(
        "--manifest", help="Reviewed selected Open Images source manifest"
    )
    args = parser.parse_args()
    import gzip
    import hashlib
    import json
    import pathlib
    import struct
    import time

    import numpy as np
    from PIL import Image, ImageDraw

    from local_gan.model import sample, train

    root = pathlib.Path(args.data)
    raw = (root / "train-images.gz").read_bytes()
    assert hashlib.md5(raw).hexdigest() == "8d4fb7e6c68d591d4c3dfef9ec88bf0d"
    b = gzip.decompress(raw)
    magic, count, rows, cols = struct.unpack(">IIII", b[:16])
    assert (magic, count, rows, cols) == (2051, 60000, 28, 28)
    images = np.frombuffer(b[16:], dtype=np.uint8).reshape(count, rows, cols)
    folder = root / "train64"
    folder.mkdir()
    for i, a in enumerate(images[:128]):
        Image.fromarray(a).convert("RGB").resize((64, 64)).save(folder / f"{i:04d}.png")
    start = time.time()
    scores = train(
        folder,
        root / "model",
        epochs=3,
        batch_size=16,
        latent=32,
        width=8,
        max_steps=24,
    )
    paths = sample(root / "model/checkpoint.pt", root / "samples", 8)
    report = {
        "real_data": True,
        "dataset": "Fashion-MNIST",
        "subset": (
            "first 128 official training images, 28x28 grayscale "
            "converted RGB/resized64; not faces"
        ),
        "license": "MIT, Zalando SE 2017",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "md5": hashlib.md5(raw).hexdigest(),
        "training": scores,
        "elapsed_seconds": round(time.time() - start, 2),
        "quality_claim": False,
        "limits": (
            "24 CPU optimizer steps, no FID/diversity/memorization or "
            "held-out quality evaluation"
        ),
        "samples": len(paths),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    canvas = Image.new("RGB", (8 * 96, 230), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), "Real Fashion-MNIST training inputs (garments)", fill="black")
    for i in range(8):
        canvas.paste(Image.open(folder / f"{i:04d}.png").resize((80, 80)), (i * 96, 25))
    d.text((10, 120), "24-step CPU DCGAN outputs, not quality evidence", fill="black")
    for i, p in enumerate(paths):
        canvas.paste(Image.open(p).resize((80, 80)), (i * 96, 145))
    canvas.save(str(root / "real-fashion-evidence.png"))


if __name__ == "__main__":
    main()
