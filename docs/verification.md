# Verification record

## Local commands actually run

- `uv sync --locked --extra dev --extra ml`: installs pinned dependencies in Python 3.11.18. The Linux PyPI Torch wheel includes CUDA libraries, but this package explicitly runs models/tensors on CPU. No GPU/paid compute was used.
- `uv run pytest --cov=local_gan --cov-report=term-missing`: 23 passed, 99% package coverage. Tests verify both adversaries change weights, finite tiny-step losses, 64x64 outputs, [-1,1] tensors, checkpoint reload/deterministic PNGs, and unchanged generator buffers/mode during sampling.
- `uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run mypy src`, `uv run pre-commit run --all-files`: required before publication. See CI for remote results.
- `uv run python scripts/synthetic_demo.py`, `uv run dcgan check-data --folder synthetic-demo/images`: 3 decoded inputs, shape [3,64,64].
- `uv run dcgan train synthetic-demo/images synthetic-demo/trained --epochs 1 --batch-size 2 --latent 3 --width 2 --max-steps 1`: one actual generator/discriminator update, losses .8501 and 1.6462 on this run, checkpoint saved locally.
- `uv run dcgan sample synthetic-demo/trained/checkpoint.pt synthetic-demo/samples --count 2`: two PNGs written. These noise-like images do not establish visual quality.
- Console and module `--help` passed. Missing required folder exits 2. Optional `.env`/GAN_IMAGES support is tested.
- Separate `/tmp/gan-light` base+dev installation has no Torch: 16 passed, three explicit ML skips (two CLI tests and backend module).

## Behavior changes and caveats

Flat-folder image loading replaces ImageFolder's class-directory assumption. Both network widths honor the supplied width; original notebook declared ndf=128 but silently used 64. Sampling now uses eval mode and restores mode, preventing BatchNorm buffer changes. Old notebook output/credential/download cells are removed or disabled; no full notebook rerun. Center cropping uses Pillow ImageOps.fit, so it is not promised pixel-identical to torchvision's previous Resize/CenterCrop. The package explicitly uses CPU, drops singleton batch tails and adds a total step cap. Existing weights are not guaranteed compatible. Checkpoints contain tensor weights/configuration, not a resumable optimizer state.

Synthetic tests use width=2 and latent=3, not full default-width real-data training. There is no real dataset benchmark, FID, train/test split, diversity/fairness/privacy assessment or proof of useful image generation. Losses are optimization diagnostics, not quality scores. Only use images you have permission to train on. Never load untrusted checkpoints, even with restricted weight loading.

## CI and container scope

The workflow runs the same lint/type/test/hooks/CLI commands, then builds and smoke-runs a data-only non-root Docker image. Hosted success must be checked on the published run before reporting Docker as verified. It does not train a full GAN or publish an image/model. No paid APIs or deployments are configured. Private-repository runs are guarded off to respect the no-fees request.

Main is not merged, PR words remain for review, and no release tag or license change is made.


## Gap-closure follow-up (24 November 2026)

The CI source now uses uv 0.13.8 and the committed transitive lock, rather than a new pip resolution of direct pins. Dependency installer caching is keyed by uv.lock. The local matching command is `uv sync --locked --extra dev --extra ml`, followed by the existing tests/quality commands. This does not cache or publish private datasets, weights or model outputs. Public-repository-only guards remain. The changed workflow's hosted outcome must be checked after publication; prior green runs do not validate a changed workflow.

A complete copy-only project-focused profile README draft is appended to presentation.md. No profile account edits, main merge, PR, release or license change is implied by that text. Historical notebook/full-data/quality limitations remain open, not replaced by a passing synthetic test.
