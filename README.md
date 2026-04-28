# TriFuse-LFCNN OPM-MEG Brain Fingerprinting

Anonymous code release for:

**TriFuse-LFCNN: Feasibility Analysis of OPM-MEG as an Emerging Brain-Fingerprinting Modality**

This repository reproduces the main OPM-MEG subject-identification experiments, compact TriFuse-LFCNN model, appendix deep baselines, and confound-aware controls used in the manuscript. The intended claim is deliberately narrow: a small public same-session OPM-MEG dataset contains strong subject cues, but sensor-space success is heavily geometry-confounded and should be interpreted as feasibility evidence, not deployment-ready biometrics.

## Repository Layout

```text
trifuse-lfcnn-opm-meg/
  scripts/      Reproduction scripts
  results/      Sanitized result summaries
  docs/         Reproducibility notes
```

Raw data, preprocessing caches, and model checkpoints are not included.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if you plan to reproduce the deep-model results on GPU.

## Data

Download the public `dog_day_afternoon_OPM` archive and either place it at:

```text
data/dog_day_afternoon_OPM.zip
```

or pass it explicitly:

```bash
python scripts/prepare_opm_movie_dataset.py --zip-path /path/to/dog_day_afternoon_OPM.zip
```

You can also set:

```bash
export OPM_MEG_ZIP=/path/to/dog_day_afternoon_OPM.zip
export OPM_MEG_CACHE=cache/opm_movie_validation_cache
```

## Quick Reproduction

Build the beta-band cache and primary classical result:

```bash
python scripts/prepare_opm_movie_dataset.py \
  --subjects 1 2 3 4 5 6 7 8 9 10 \
  --band beta \
  --modes logvar_lr \
  --output-json results/primary_logvar_beta.json
```

Run TriFuse-LFCNN and the no-gate LF-CNN ablation:

```bash
python scripts/run_trifuse_lfcnn.py \
  --configs trifuse_lfcnn lfcnn_concat_ablation \
  --output-json results/trifuse_lfcnn_beta.json
```

Run TriFuse embedding-level biometric controls, multiband rejection, and geometry-residualized embedding diagnostics:

```bash
python scripts/run_trifuse_biometric_controls.py \
  --output-dir results
```

Run appendix deep baselines:

```bash
python scripts/run_deep_baselines.py \
  --models eegnet shallowconvnet deepconvnet eegconformer \
  --output-json results/deep_baseline_benchmark_beta.json
```

Run the main confound controls:

```bash
python scripts/run_sensor_confound_controls.py
python scripts/run_common_grid_controls.py
python scripts/run_anatomical_source_grid_controls.py
python scripts/run_geometry_residualized_controls.py
```

Run high-confidence and open-set diagnostics:

```bash
python scripts/run_open_set_operating_points.py
python scripts/run_multiband_confidence.py
```

## Expected Headline Results

The sanitized summary is in `results/headline_results.json`. Main numbers:

- Primary beta LogVar + logistic regression: 99.42% closed-set accuracy, 3.30% EER.
- Subject-clustered 95% CI for primary accuracy: 98.98% to 99.74%.
- TriFuse-LFCNN: 99.60% mean cross-run accuracy with 490,298 trainable parameters.
- No-gate LF-CNN ablation: 99.29% mean cross-run accuracy.
- TriFuse beta fused-embedding verification: 0.43% EER under run-1 enrollment and run-2 probe.
- TriFuse strict multiband rejection: 86.13% to 89.71% unknown rejection across 1-3 unknown-subject settings, with about 99.0% known-subject accepted accuracy.
- Geometry metadata alone: 100.00% accuracy, showing severe sensor-geometry leakage risk.
- Geometry-residualized LogVar collapses near chance: 9.20% native sensor-space accuracy and 10.55% common-grid accuracy.
- Geometry-residualized TriFuse embeddings also collapse near chance: 10.10% mean accuracy and 58.91% mean EER.

These results should be read together: the benchmark contains strong subject cues and TriFuse provides a strong compact embedding, but raw sensor-space performance is not sufficient evidence for pure neural permanence. The embedding residualization result supports a geometry-neural entanglement interpretation rather than a pure-neural claim.

## Scripts

- `scripts/prepare_opm_movie_dataset.py`: cache construction and primary LogVar/LR validation.
- `scripts/run_trifuse_lfcnn.py`: TriFuse-LFCNN and no-gate LF-CNN ablation.
- `scripts/run_trifuse_biometric_controls.py`: TriFuse fused-embedding verification, open-set diagnostics, multiband rejection, common-grid time-series control, and geometry-residualized embedding controls.
- `scripts/run_deep_baselines.py`: EEGNet, ShallowConvNet, DeepConvNet, and EEG Conformer baselines.
- `scripts/run_sensor_confound_controls.py`: native sensor-space and geometry-only controls.
- `scripts/run_common_grid_controls.py`: world-coordinate common-grid controls.
- `scripts/run_anatomical_source_grid_controls.py`: anatomy-aware source-anchor proxy controls.
- `scripts/run_geometry_residualized_controls.py`: train-run-only geometry residualization controls.
- `scripts/run_logeuclidean_covariance.py`: log-Euclidean covariance baseline.
- `scripts/run_open_set_operating_points.py`: open-set operating-point diagnostics.
- `scripts/run_multiband_confidence.py`: multiband high-confidence rejection diagnostics.
- `scripts/run_subject_clustered_uncertainty.py`: subject-clustered bootstrap uncertainty for the primary result.

## Notes

The code is arranged for anonymous review. It avoids local absolute paths and does not include raw datasets or private artifacts. See `docs/REPRODUCIBILITY.md` for protocol details. The release uses an anonymous-author MIT license placeholder that can be updated after review.
