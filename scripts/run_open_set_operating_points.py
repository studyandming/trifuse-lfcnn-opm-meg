import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

from release_utils import default_cache_dir, default_result_path

CACHE_DIR = default_cache_dir()
NPZ_NAME = (
    "sub-001_sub-002_sub-003_sub-004_sub-005_sub-006_"
    "sub-007_sub-008_sub-009_sub-010_beta_200hz_5s.npz"
)
OUT_PATH = default_result_path("open_set_operating_points_beta.json")


def load_beta_features():
    data = np.load(CACHE_DIR / NPZ_NAME, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    run_ids = data["run_ids"].astype(np.int64)
    features = np.log(np.var(x, axis=-1) + 1e-6).reshape(x.shape[0], -1).astype(np.float32)
    return features, y, run_ids


def dir_at_fixed_fpir(features, y, run_ids, n_unknown=2, n_repeats=20, fpir_targets=(0.01, 0.05, 0.10)):
    subjects = np.unique(y)
    all_results = {target: [] for target in fpir_targets}

    for rep in range(n_repeats):
        np.random.seed(rep)
        unknown_subs = np.random.choice(subjects, n_unknown, replace=False)
        known_subs = [subject for subject in subjects if subject not in unknown_subs]

        enroll_idx = np.where(run_ids == 1)[0]
        probe_idx = np.where(run_ids == 2)[0]

        templates = {}
        for subject in known_subs:
            mask = y[enroll_idx] == subject
            templates[subject] = features[enroll_idx[mask]].mean(axis=0)

        known_best_scores = []
        known_correct = []
        unknown_best_scores = []

        for idx in probe_idx:
            probe_feat = features[idx]
            true_id = y[idx]
            best_score = -1.0
            best_id = -1
            for subject, tmpl in templates.items():
                score = 1.0 - cosine(probe_feat, tmpl)
                if score > best_score:
                    best_score = score
                    best_id = subject

            if true_id in unknown_subs:
                unknown_best_scores.append(best_score)
            else:
                known_best_scores.append(best_score)
                known_correct.append(1 if best_id == true_id else 0)

        known_best_scores = np.asarray(known_best_scores, dtype=np.float64)
        known_correct = np.asarray(known_correct, dtype=np.int64)
        unknown_best_scores = np.asarray(unknown_best_scores, dtype=np.float64)

        thresholds = np.unique(np.concatenate([known_best_scores, unknown_best_scores]))
        thresholds = np.concatenate(
            ([thresholds.min() - 1e-6], thresholds, [thresholds.max() + 1e-6])
        )

        fpirs = []
        dirs = []
        for threshold in thresholds:
            fpir = float(np.mean(unknown_best_scores >= threshold))
            dir_value = float(np.mean((known_best_scores >= threshold) & (known_correct == 1)))
            fpirs.append(fpir)
            dirs.append(dir_value)

        fpirs = np.asarray(fpirs, dtype=np.float64)
        dirs = np.asarray(dirs, dtype=np.float64)

        for target in fpir_targets:
            valid = fpirs <= target
            best_dir = float(np.max(dirs[valid])) if np.any(valid) else 0.0
            all_results[target].append(best_dir)

    summary = {}
    for target, values in all_results.items():
        key = f"dir_at_fpir_{int(target * 100)}"
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


def main():
    global CACHE_DIR, OUT_PATH
    parser = argparse.ArgumentParser(description="Open-set operating-point diagnostics for OPM-MEG subject identification.")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("open_set_operating_points_beta.json"))
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    OUT_PATH = args.output_json

    features, y, run_ids = load_beta_features()
    results = {}
    for n_unknown in (1, 2, 3):
        results[f"{n_unknown}unk"] = dir_at_fixed_fpir(
            features, y, run_ids, n_unknown=n_unknown
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
