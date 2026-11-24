# Real-data CPU development run

Source: https://github.com/zalandoresearch/fashion-mnist . Actual recorded result: `real-data-output.json`. This is a small development run, not full training, production readiness or an untouched benchmark. No data/model artifacts are committed. Run from the repository root after installing the locked ML/dev extras. This script starts real CPU training only when explicitly invoked. Outputs use fixed new names under the supplied data folder and refuse rerunning into an existing artifact. Use a new folder for another run.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/real_data.py --data /path/to/prepared-source
```

Prepare the official `data/fashion/train-images-idx3-ubyte.gz` as `train-images.gz`. Repository source MIT notice, Zalando SE 2017, must be retained with copies. The script checks published MD5 8d4fb7e6c68d591d4c3dfef9ec88bf0d. It uses first128 official training garments, resized64 RGB, trains 24 small CPU DCGAN steps and samples8. They are clothes, not faces. No FID/diversity/privacy score. Official files: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion . About26MB compressed; installed ML environment is larger.

## Actual limits

24 CPU optimizer steps, no FID/diversity/memorization or held-out quality evaluation

## Explicit download and preparation

The optional helper downloads only the documented primary-source files, checks SHA-256 and refuses an existing destination. It never runs during installation, import or tests. Source attribution/license cautions above still apply. No model/data is bundled. For an offline copy, add `--cached /path/to/source-files` with the same filenames; checksums are still required.

```sh
uv run --no-sync python scripts/fetch_real_data.py --output ./real-source
uv run --no-sync python scripts/real_data.py --data ./real-source
```

The hashes pin the exact small development sources used here, not a permission warranty or a quality benchmark. If a source changes, the helper fails rather than silently training on different data.
