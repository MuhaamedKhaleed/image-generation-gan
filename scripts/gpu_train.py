"""Explicit GPU/CPU Fashion-MNIST training at a real scale. Never runs on import.

Uses the package network topology from local_gan.model with an explicit device
so free Kaggle/Colab GPUs work. Saves the same format-1 checkpoint that
`dcgan sample` loads, plus a report and an evidence grid. This is a larger
engineering run, not a face-quality, FID or diversity claim.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Folder from fetch_real_data.py")
    parser.add_argument("--output", required=True, help="New output folder")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:0")
    parser.add_argument(
        "--images", type=int, default=60000, help="Leading training images used"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent", type=int, default=100)
    parser.add_argument("--width", type=int, default=128)
    # Canonical DCGAN rate. The repository default 0.0004 was observed to
    # collapse at this scale: discriminator loss reached 0.0 by step 100 and
    # the generator received no useful gradient (Kaggle T4 run, 2026-11-25).
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import gzip
    import hashlib
    import json
    import struct
    import time
    from pathlib import Path

    root = Path(args.data)
    out = Path(args.output)
    if out.exists():
        parser.error("Output exists; use a new directory.")
    if not 1 <= args.images <= 60000:
        parser.error("--images must be within 1..60000 (official train count).")

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch import nn
    from torch.optim.adam import Adam

    from local_gan.model import networks, sample

    raw = (root / "train-images.gz").read_bytes()
    if hashlib.md5(raw).hexdigest() != "8d4fb7e6c68d591d4c3dfef9ec88bf0d":
        parser.error(
            "train-images.gz MD5 mismatch; run scripts/fetch_real_data.py first."
        )
    b = gzip.decompress(raw)
    magic, count, rows, cols = struct.unpack(">IIII", b[:16])
    assert (magic, count, rows, cols) == (2051, 60000, 28, 28)
    images = np.frombuffer(b[16:], dtype=np.uint8).reshape(count, rows, cols)[
        : args.images
    ]
    # Match core.image_array preprocessing: RGB, bilinear 64x64, [-1, 1].
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(a).convert("RGB").resize((64, 64), Image.BILINEAR),
                dtype=np.float32,
            )
            for a in images
        ]
    )
    data = torch.from_numpy(resized.transpose(0, 3, 1, 2) / 127.5 - 1.0)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")
    torch.manual_seed(args.seed)
    g, d = networks(args.latent, args.width)
    g.to(device)
    d.to(device)
    og = Adam(g.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    od = Adam(d.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    loss = nn.BCELoss()
    steps = 0
    start = time.time()
    history = []
    for _epoch in range(args.epochs):
        order = torch.randperm(len(data)).tolist()
        for i in range(0, len(data), args.batch_size):
            idx = order[i : i + args.batch_size]
            if len(idx) < 2:
                continue
            real = data[idx].to(device)
            noise = torch.randn(len(idx), args.latent, 1, 1, device=device)
            fake = g(noise)
            od.zero_grad(set_to_none=True)
            dl = loss(d(real), torch.ones(len(idx), device=device)) + loss(
                d(fake.detach()), torch.zeros(len(idx), device=device)
            )
            dl.backward()
            od.step()
            og.zero_grad(set_to_none=True)
            gl = loss(d(fake), torch.ones(len(idx), device=device))
            if not bool(torch.isfinite(gl)) or not bool(torch.isfinite(dl)):
                raise SystemExit("Nonfinite training loss; no checkpoint saved.")
            gl.backward()
            og.step()
            steps += 1
            if steps % 100 == 0 or steps == 1:
                row = {
                    "step": steps,
                    "generator_loss": round(float(gl.detach().cpu()), 4),
                    "discriminator_loss": round(float(dl.detach().cpu()), 4),
                }
                history.append(row)
                print(row, flush=True)
            if steps >= args.max_steps:
                break
        if steps >= args.max_steps:
            break

    out.mkdir(parents=True)
    g.cpu()
    d.cpu()
    torch.save(
        {
            "format": 1,
            "latent": args.latent,
            "width": args.width,
            "generator": g.state_dict(),
            "discriminator": d.state_dict(),
            "seed": args.seed,
            "steps": steps,
        },
        out / "checkpoint.pt",
    )
    paths = sample(out / "checkpoint.pt", out / "samples", 16, args.seed)
    report = {
        "real_data": True,
        "dataset": "Fashion-MNIST",
        "subset": (
            f"first {len(data)} official training garments resized 64x64 RGB; not faces"
        ),
        "license": "MIT, Zalando SE 2017",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "device": str(device),
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent": args.latent,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "steps": steps,
        "elapsed_seconds": round(time.time() - start, 2),
        "history": history,
        "samples": len(paths),
        "checkpoint_sha256": hashlib.sha256(
            (out / "checkpoint.pt").read_bytes()
        ).hexdigest(),
        "run_status": (
            "Run record only, not a quality assertion. Inspect evidence-grid.png "
            "to judge the samples; no FID, diversity or memorization score is "
            "measured."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    canvas = Image.new("RGB", (8 * 84, 4 * 84 + 60), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), "Real Fashion-MNIST training inputs (garments)", fill="black")
    for i in range(16):
        tile = Image.fromarray(images[i]).convert("RGB").resize((80, 80))
        canvas.paste(tile, ((i % 8) * 84, 24 + (i // 8) * 84))
    draw.text((8, 24 + 2 * 84 + 4), "DCGAN samples after training", fill="black")
    for i, p in enumerate(paths):
        canvas.paste(
            Image.open(p).resize((80, 80)),
            ((i % 8) * 84, 24 + 2 * 84 + 24 + (i // 8) * 84),
        )
    canvas.save(str(out / "evidence-grid.png"))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
