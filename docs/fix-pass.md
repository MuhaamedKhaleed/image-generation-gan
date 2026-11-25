# Provenance follow-up, 25 November 2026

Training now saves provenance.json with input-image SHA256 hashes, network settings, actual optimizer steps and checkpoint hash. Sampling writes its own checkpoint hash, seed, count and output PNG hashes. These local manifests tie newly produced samples to the actual run; they do not retroactively authenticate historical notebook PNGs or establish perceptual quality, diversity or lack of memorization. Keep manifests private when filenames could disclose personal information. No weights/images are committed.

Nonfinite adversarial losses and sample tensors are refused. Hashes detect changes, not malicious inputs or license permission. Existing checkpoints without a manifest can still sample, but their original training provenance remains unknown. Checkpoints lack optimizer state, so exact resume is still not offered. Original face objective is unchanged; Fashion-MNIST remains a separate engineering example.

The original CelebA terms allow noncommercial research and prohibit dataset redistribution. Terms verified at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html . No source face gallery/model newly published. Original generator weights have not been recovered from the repository or available Kaggle model listing. No face-quality claim added.
