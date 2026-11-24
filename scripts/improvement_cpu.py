"""Explicit local development experiment. No downloads or import-time work."""


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory.")
    """Explicit native-resolution MLP GAN alternative, no downloads."""
    import json
    import time
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from torch import nn

    out = args.output
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    x = torch.from_numpy(
        np.stack(
            [
                np.asarray(
                    Image.open(p).convert("L").resize((28, 28)), dtype=np.float32
                ).reshape(-1)
                / 127.5
                - 1
                for p in sorted(args.data.glob("*.png"))
            ]
        )
    )
    g = nn.Sequential(
        nn.Linear(64, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 784),
        nn.Tanh(),
    )
    d = nn.Sequential(
        nn.Linear(784, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    og = torch.optim.Adam(g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    od = torch.optim.Adam(d.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.BCEWithLogitsLoss()
    fixed = torch.randn(64, 64, generator=torch.Generator().manual_seed(31415))
    history = []
    start = time.time()
    samples = []

    def snap(step):
        g.eval()
        with torch.no_grad():
            a = g(fixed).reshape(64, 28, 28).numpy()
        g.train()
        samples.append((step, a))

    snap(0)
    for step in range(1, 3001):
        real = x[torch.randint(len(x), (64,))]
        z = torch.randn(64, 64)
        fake = g(z)
        od.zero_grad()
        dl = loss(d(real), torch.ones(64, 1)) + loss(
            d(fake.detach()), torch.zeros(64, 1)
        )
        dl.backward()
        od.step()
        og.zero_grad()
        for p in d.parameters():
            p.requires_grad_(False)
        gl = loss(d(fake), torch.ones(64, 1))
        gl.backward()
        og.step()
        for p in d.parameters():
            p.requires_grad_(True)
        if step in [250, 500, 1000, 2000, 3000]:
            snap(step)
            r = dict(
                step=step,
                g_loss=float(gl.detach()),
                d_loss=float(dl.detach()),
                seconds=time.time() - start,
            )
            history.append(r)
            print(r, flush=True)
    canvas = Image.new("L", (28 * 16, 28 * 4 * len(samples)))
    for row, (_step, a) in enumerate(samples):
        for i, im in enumerate(a):
            canvas.paste(
                Image.fromarray(((im.clip(-1, 1) + 1) * 127.5).astype("uint8")),
                (i % 16 * 28, (row * 4 + i // 16) * 28),
            )
    canvas.resize((896, 224 * len(samples))).save(out / "all-fixed.png")
    torch.save(g.state_dict(), out / "generator.pt")
    (out / "report.json").write_text(
        json.dumps(
            dict(
                model=(
                    "MLP GAN at28x28 grayscale, explicit alternative not controlled "
                    "single-change DCGAN fix "
                ),
                train_images=len(x),
                steps=3000,
                seed=42,
                fixed_seed=31415,
                history=history,
                elapsed_seconds=time.time() - start,
                limits=(
                    "one seed, training data only, 64 fixed latents at every shown "
                    "checkpoint, no FID or realism metric; inspect pixels before "
                    "claims "
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
