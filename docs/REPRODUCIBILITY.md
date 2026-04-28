# Reproducibility Notes

This release is organized for anonymous review. It does not include raw data, preprocessed caches, trained checkpoints, or local machine paths.

## Data

Download the public `dog_day_afternoon_OPM` archive and either place it at:

```text
data/dog_day_afternoon_OPM.zip
```

or point scripts to it with:

```bash
export OPM_MEG_ZIP=/path/to/dog_day_afternoon_OPM.zip
```

On Windows PowerShell:

```powershell
$env:OPM_MEG_ZIP = "D:\path\to\dog_day_afternoon_OPM.zip"
```

## Protocol

All main scripts use the same bidirectional cross-run protocol:

```text
run-1 train / validation -> run-2 test
run-2 train / validation -> run-1 test
report the mean over both directions
```

Deep models select checkpoints using a stratified validation split from the training run only. Test-run windows are not used for checkpoint selection.

## Recommended Reproduction Order

1. Build the beta cache and primary LogVar result.
2. Run TriFuse-LFCNN and the no-gate LF-CNN ablation.
3. Run TriFuse fused-embedding biometric controls, geometry-residualized embedding controls, common-grid TriFuse, and strict multiband rejection.
4. Run deep baseline models for appendix comparisons.
5. Run geometry, common-grid, anatomy-aware, and residualized LogVar confound controls.
6. Run open-set and multiband high-confidence LogVar diagnostics.

The TriFuse biometric-control script writes four manuscript-facing JSON files:

```text
results/trifuse_biometric_controls_beta.json
results/trifuse_multiband_open_set.json
results/trifuse_geometry_residualized_embeddings.json
results/trifuse_common_grid_beta.json
```

The cache directory defaults to `cache/opm_movie_validation_cache`. Set `OPM_MEG_CACHE` or pass `--cache-dir` to use another location.
