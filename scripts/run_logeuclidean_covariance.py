import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_utils import default_cache_dir, default_result_path

CACHE_DIR = default_cache_dir()
OUT_PATH = default_result_path("logeuclidean_covariance_beta.json")


def load_cached_beta() -> Dict:
    tag = "_".join([f"sub-{i:03d}" for i in range(1, 11)])
    npz_path = CACHE_DIR / f"{tag}_beta_200hz_5s.npz"
    json_path = CACHE_DIR / f"{tag}_beta_200hz_5s.json"
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "subject_ids": data["subject_ids"].astype(np.int64),
        "run_ids": data["run_ids"].astype(np.int64),
        "meta": meta,
    }


def upper_triangle_with_metric_weight(mats: np.ndarray) -> np.ndarray:
    n_chan = mats.shape[1]
    tri_i, tri_j = np.triu_indices(n_chan)
    feat = mats[:, tri_i, tri_j].astype(np.float32)
    off_diag = tri_i != tri_j
    feat[:, off_diag] *= np.sqrt(2.0, dtype=np.float32)
    return feat


def logeuclidean_cov_features(
    x: np.ndarray,
    reg: float = 1e-3,
    batch_size: int = 64,
) -> np.ndarray:
    n_samples, n_axes, n_sensors, n_time = x.shape
    n_chan = n_axes * n_sensors
    features = []

    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        xb = x[start:stop].reshape(stop - start, n_chan, n_time).astype(np.float64)
        xb = xb - xb.mean(axis=-1, keepdims=True)

        cov = np.einsum("nct,ndt->ncd", xb, xb, optimize=True) / max(n_time - 1, 1)
        trace_scale = np.trace(cov, axis1=1, axis2=2) / n_chan
        cov[:, np.arange(n_chan), np.arange(n_chan)] += reg * trace_scale[:, None]

        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-9, None)
        log_cov = np.einsum("nij,nj,nkj->nik", evecs, np.log(evals), evecs, optimize=True)
        features.append(upper_triangle_with_metric_weight(log_cov))

    return np.concatenate(features, axis=0).astype(np.float32)


def cross_run_accuracy(features: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Dict:
    folds = []
    for train_run, test_run in [(1, 2), (2, 1)]:
        tr = run_ids == train_run
        te = run_ids == test_run
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(features[tr], y[tr])
        pred = clf.predict(features[te])
        folds.append(float(accuracy_score(y[te], pred)))
    return {
        "folds": folds,
        "mean_acc": float(np.mean(folds)),
        "std_acc": float(np.std(folds)),
    }


def verification_scores(features: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    enroll_idx = np.where(run_ids == 1)[0]
    probe_idx = np.where(run_ids == 2)[0]
    subjects = np.unique(y)
    templates = {}
    for s in subjects:
        templates[int(s)] = features[enroll_idx[y[enroll_idx] == s]].mean(axis=0)

    genuine, impostor = [], []
    for idx in probe_idx:
        feat = features[idx]
        true_id = int(y[idx])
        for s in subjects:
            score = 1.0 - cosine(feat, templates[int(s)])
            if int(s) == true_id:
                genuine.append(score)
            else:
                impostor.append(score)
    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def main() -> None:
    global CACHE_DIR, OUT_PATH
    parser = argparse.ArgumentParser(description="Log-Euclidean covariance baseline under the IJCB cross-run protocol.")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("logeuclidean_covariance_beta.json"))
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    OUT_PATH = args.output_json

    data = load_cached_beta()
    features = logeuclidean_cov_features(data["X"])
    acc = cross_run_accuracy(features, data["y"], data["run_ids"])
    genuine, impostor = verification_scores(features, data["y"], data["run_ids"])

    out = {
        "dataset": "dog_day_afternoon_OPM",
        "band": "beta",
        "window_s": 5,
        "n_subjects": 10,
        "n_common_sensors": int(len(data["meta"]["common_sensors"])),
        "feature_family": "log-euclidean covariance",
        "input_channels": 147,
        "feature_dim": int(features.shape[1]),
        "covariance_regularization": "trace-scaled diagonal loading, reg=1e-3",
        "folds": acc["folds"],
        "mean_acc": acc["mean_acc"],
        "std_acc": acc["std_acc"],
        "eer": compute_eer(genuine, impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
