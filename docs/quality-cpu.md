# Measured no-cost CPU quality experiment

2048FashionMNISTgarments,1000optimizersteps,width16,latent64,113CPU seconds419MiB. Inspected16samples remain noise. Biggertraining alone did not yield visualquality. visual-panel.png top2rows generated,bottom2rows realinputsoutside selected2048. NoFID/diversity/memorization claim. Output is a failedqualityexperiment, not completion of usefulgeneration.

Run only explicitly, with previously prepared local permitted data and a new output. No automatic downloads or import-time execution. Exact measured result is quality-output.json. This is a reproducible development experiment, not a productionquality guarantee.

```sh
uv sync --locked --extra dev --extra ml
uv run --no-sync python scripts/quality_cpu.py --data LOCAL_PREPARED_SOURCE --output NEW_OUTPUT
```

Test sets described as finalized have now been observed. Future choices must not be justified with their scores as if they remained unseen. Retain source terms/attribution from real-data.md. No model or raw text/photos committed.
