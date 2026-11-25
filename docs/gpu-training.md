# Larger GPU training (Kaggle/Colab)

The 24-step CPU wiring demo proves the loop runs; it does not produce
recognizable garments. `scripts/gpu_train.py` trains the same package DCGAN
topology (`local_gan.model.networks`) on the full official Fashion-MNIST
training set with an explicit `--device`, so a free Kaggle or Colab GPU works.
Fashion-MNIST is MIT-licensed (Zalando SE 2017); it is garments, not faces.

## Run on a fresh clone (free Kaggle/Colab GPU)

```sh
python scripts/fetch_real_data.py --output ./real-source
python scripts/gpu_train.py --data ./real-source --output ./gpu-out --device cuda --images 60000 --epochs 3 --batch-size 128
```

Use `--device cpu` locally; the CPU path was verified on a tiny run. The script
saves a format-1 checkpoint that `dcgan sample` loads unchanged, plus
`report.json` and `evidence-grid.png` (real inputs vs generated samples).
On Kaggle/Colab, install nothing pinned to CPU: clone the branch, put `src` on
`PYTHONPATH` (the locked environment targets Linux x86_64 CPU torch), and use
the runtime's preinstalled CUDA torch.

## Actual recorded result

Kaggle GPU run `gan-kaggle-run` (private notebook in the owner's Kaggle
account, GPU T4 x2, torch 2.11.0+cu130, CUDA verified available):

| attempt | learning rate | outcome |
| --- | --- | --- |
| 2026-11-25 13:47 EET | 0.0004 (repo default) | Discriminator collapse: discriminator loss reached 0.0 by step 100 and stayed there, generator loss flat near 51 with no useful gradient. Run stopped; no checkpoint kept. |
| 2026-11-25 14:21 EET | 0.0002 (canonical DCGAN rate, now the script default) | Same collapse at the lower rate: 1,407 steps (3 epochs, 60,000 images, batch 128) in 864.5s on a Tesla T4 (CUDA verified, seed 42). Discriminator loss 0.0 by step 100, generator loss flat near 51-61; the evidence grid is pure noise, no garment structure. Checkpoint kept as failure evidence (sha256 ba0a706ea0bb731a8b84de37e5f54a6a69624a8ebdcad70442d1d205bed0e173). No third tuning attempt without coordination. |

Both full-data GPU attempts collapsed. This is failed-training evidence, not
a working model: the verified outputs remain the local CPU tiny-run (wiring)
and the 24-step demo. `report.json` carries a neutral `run_status` field (a
run record, not a quality assertion); judge `evidence-grid.png` directly.

## Face generation options (research, not adopted)

The original notebook targeted faces; its weights were not recovered. Checked
options and why none is bundled:

- CelebA dataset: non-commercial research use only, requires agreement;
  redistribution restricted. Suitable for a personal research rerun, not for a
  committed artifact.
- `huggan/dcgan-celeba-faces` pretrained weights: no license field on the
  model card (checked via the Hugging Face API), so not adopted.
- Other public CelebA DCGAN uploads are personal repositories with unverified
  training-data provenance.

Do not use generated faces to impersonate real people.
