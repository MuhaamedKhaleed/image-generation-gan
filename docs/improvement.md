# Measured development alternative, 24 November 2026

Native 28x28 grayscale MLP GAN, not the original DCGAN. All64 fixed outputs at0/250/500/1000/2000/3000 inspected: rough garment silhouettes, still grainy/distorted. No FID or untouched quality evaluation. Pixel nearest-train checks are not proof against memorization. Input is the previously prepared2048 PNGs, resized to28x28. Model checkpoint stays local.

Only explicit local execution. No download/import side effects. Use a new output directory.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/improvement_cpu.py --data LOCAL_SOURCE --output NEW_OUTPUT
```

The portable script was rerun and reproduced the recorded metrics. Environment Python3.11 Linux x86_64 CPU. Raw data and trained weights stay outside GitHub. Original baseline and limitations remain in quality-cpu.md.

```json
{
  "model": "MLP GAN at28x28 grayscale, explicit alternative not controlled single-change DCGAN fix",
  "train_images": 2048,
  "steps": 3000,
  "seed": 42,
  "fixed_seed": 31415,
  "history": [
    {
      "step": 250,
      "g_loss": 0.7589324712753296,
      "d_loss": 1.2490172386169434,
      "seconds": 2.2273502349853516
    },
    {
      "step": 500,
      "g_loss": 0.8248149752616882,
      "d_loss": 1.2507569789886475,
      "seconds": 4.131504058837891
    },
    {
      "step": 1000,
      "g_loss": 0.9398003816604614,
      "d_loss": 1.2950266599655151,
      "seconds": 7.920258045196533
    },
    {
      "step": 2000,
      "g_loss": 0.8372382521629333,
      "d_loss": 1.3103214502334595,
      "seconds": 15.570350646972656
    },
    {
      "step": 3000,
      "g_loss": 0.807204008102417,
      "d_loss": 1.3751845359802246,
      "seconds": 23.20539426803589
    }
  ],
  "elapsed_seconds": 23.296349048614502,
  "limits": "one seed, training data only, 64 fixed latents at every shown checkpoint, no FID or realism metric; inspect pixels before claims",
  "portable_rerun_verified": true
}
```
