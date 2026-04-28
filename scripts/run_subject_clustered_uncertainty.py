import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_opm_movie_dataset import (
    prepare_dataset,
    stratified_train_val_indices,
    subject_token,
)
from release_utils import default_cache_dir, default_result_path, default_zip_path


ZIP_PATH = default_zip_path()
CACHE_DIR = default_cache_dir()
OUT_PATH = default_result_path("subject_clustered_uncertainty_beta.json")


def make_logvar_features(x: np.ndarray) -> np.ndarray:
    feats = np.log(np.var(x, axis=-1) + 1e-6)
    return feats.reshape(feats.shape[0], -1).astype(np.float32, copy=False)


def run_beta_logvar_predictions() -> Dict:
    prepared = prepare_dataset(
        zip_path=ZIP_PATH,
        subject_ids=list(range(1, 11)),
        runs=[1, 2],
        band_name="beta",
        target_fs=200,
        window_s=5.0,
        movie_s=600.0,
        cache_dir=CACHE_DIR,
        force_rebuild=False,
    )
    x = prepared.X.astype(np.float32, copy=False)
    y = prepared.y
    subject_ids = prepared.subject_ids
    run_ids = prepared.run_ids
    features = make_logvar_features(x)

    folds: List[Dict] = []
    per_subject_correct = {int(s): [] for s in np.unique(subject_ids)}

    for train_run, test_run in [(1, 2), (2, 1)]:
        train_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"),
        )
        clf.fit(features[train_idx], y[train_idx])
        pred = clf.predict(features[test_idx])
        correct = pred == y[test_idx]
        for sid in np.unique(subject_ids[test_idx]):
            mask = subject_ids[test_idx] == sid
            per_subject_correct[int(sid)].extend(correct[mask].astype(float).tolist())

        folds.append(
            {
                "train_run": int(train_run),
                "test_run": int(test_run),
                "C": 1.0,
                "test_acc": float(np.mean(correct)),
                "test_windows": int(len(test_idx)),
                "correct_windows": int(np.sum(correct)),
            }
        )

    subject_items = []
    subject_accs = []
    for sid, values in sorted(per_subject_correct.items()):
        arr = np.asarray(values, dtype=float)
        acc = float(arr.mean())
        subject_accs.append(acc)
        subject_items.append(
            {
                "subject": subject_token(sid),
                "windows": int(arr.size),
                "accuracy": acc,
                "errors": int(arr.size - arr.sum()),
            }
        )

    rng = np.random.default_rng(20260427)
    subject_accs = np.asarray(subject_accs, dtype=float)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, len(subject_accs), size=len(subject_accs))
        boot.append(float(subject_accs[idx].mean()))
    boot = np.asarray(boot, dtype=float)

    return {
        "task": "subject-clustered uncertainty for primary beta LogVar+LR closed-set identification",
        "protocol": "same 10-subject beta 5s bidirectional cross-run protocol as the manuscript primary table; fixed LogReg C=1.0; CI bootstraps subjects, not windows",
        "folds": folds,
        "window_level_mean_acc": float(np.mean([f["test_acc"] for f in folds])),
        "subject_level_mean_acc": float(subject_accs.mean()),
        "subject_level_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "per_subject": subject_items,
        "bootstrap": {"n_resamples": 10000, "seed": 20260427},
    }


def main() -> None:
    global ZIP_PATH, CACHE_DIR, OUT_PATH
    parser = argparse.ArgumentParser(description="Subject-clustered uncertainty for the primary beta LogVar+LR result.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("subject_clustered_uncertainty_beta.json"))
    args = parser.parse_args()
    ZIP_PATH = args.zip_path
    CACHE_DIR = args.cache_dir
    OUT_PATH = args.output_json

    results = run_beta_logvar_predictions()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
