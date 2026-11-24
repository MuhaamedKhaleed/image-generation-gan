# Changelog

## [Unreleased]

### Fixed
- Honor the discriminator feature width and preserve generator BatchNorm buffers during sampling.
- Refuse the legacy flat-folder/ImageFolder mismatch instead of silently assuming class folders.

### Added
- Strict local image loading, bounded CPU adversarial training and restricted local checkpoint sampling.
- Synthetic model/CLI/error tests, dependency lock, quality checks and data-only container workflow.

### Changed
- Cleared notebook outputs and disabled online credential/download setup.
- Documented unmeasured real-data quality and differences from the historical notebook.
