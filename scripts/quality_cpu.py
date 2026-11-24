"Explicit measured CPU quality experiment. No import-time work or downloads."


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model", type=Path, help="Translation only: trusted local Marian"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; use a new directory.")
    import gzip
    import json
    import resource
    import time
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from local_gan.model import sample, train

    root = Path(str(args.output))
    root.mkdir(exist_ok=False)
    folder = root / "images"
    folder.mkdir()
    b = gzip.decompress((args.data / "train-images.gz").read_bytes())
    images = np.frombuffer(b[16:], dtype=np.uint8).reshape(60000, 28, 28)
    for i, a in enumerate(images[:2048]):
        Image.fromarray(a).convert("RGB").resize((64, 64)).save(folder / f"{i:04d}.png")
    start = time.time()
    r = train(
        folder,
        root / "model",
        epochs=20,
        batch_size=32,
        latent=64,
        width=16,
        max_steps=1000,
        learning_rate=0.0002,
    )
    paths = sample(root / "model/checkpoint.pt", root / "samples", 16, 42)
    r.update(
        source="https://github.com/zalandoresearch/fashion-mnist",
        training_images=2048,
        elapsed_seconds=time.time() - start,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limits=(
            "1000steps not FID or quality benchmark; no test or "
            "memorization claim. Visual sample inspection required."
        ),
    )
    (root / "report.json").write_text(json.dumps(r, indent=2))
    canvas = Image.new("RGB", (512, 256), "white")
    for i, p in enumerate(paths):
        canvas.paste(Image.open(p), (i % 8 * 64, i // 8 * 64))
    for i, a in enumerate(images[3000:3016]):
        canvas.paste(
            Image.fromarray(a).convert("RGB").resize((64, 64)),
            (i % 8 * 64, 128 + i // 8 * 64),
        )
    canvas.save(root / "visual-panel.png")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
