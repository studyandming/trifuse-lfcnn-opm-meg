import argparse
import json
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_utils import default_cache_dir, default_result_path, default_zip_path


CACHE_DIR = default_cache_dir()
ZIP_PATH = default_zip_path()
OUT_PATH = default_result_path("sensor_confound_controls_beta.json")


def load_cached_beta() -> Dict:
    tag = "_".join([f"sub-{i:03d}" for i in range(1, 11)])
    npz_path = CACHE_DIR / f"{tag}_beta_200hz_5s.npz"
    json_path = CACHE_DIR / f"{tag}_beta_200hz_5s.json"
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "run_ids": data["run_ids"].astype(np.int64),
        "meta": meta,
    }


def cross_run_accuracy(features: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Dict:
    folds: List[float] = []
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


def feat_logvar(x: np.ndarray) -> np.ndarray:
    v = np.log(np.var(x, axis=-1) + 1e-6)
    return v.reshape(v.shape[0], -1).astype(np.float32)


def feat_relative_logvar(x: np.ndarray) -> np.ndarray:
    f = feat_logvar(x)
    f = f - f.mean(axis=1, keepdims=True)
    return f.astype(np.float32)


def feat_sensor_power(x: np.ndarray) -> np.ndarray:
    # Orientation-invariant per-sensor power using the tri-axial variance norm.
    v = np.var(x, axis=-1)
    sensor_power = np.log(v.sum(axis=1) + 1e-6)
    return sensor_power.astype(np.float32)


def feat_relative_sensor_power(x: np.ndarray) -> np.ndarray:
    f = feat_sensor_power(x)
    f = f - f.mean(axis=1, keepdims=True)
    return f.astype(np.float32)


def load_geometry_features(common_sensors: List[str]) -> Dict:
    subjects = [f"sub-{i:03d}" for i in range(1, 11)]
    runs = [1, 2]

    samples = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for sub in subjects:
            sid = int(sub.split("-")[1])
            for run in runs:
                name = f"dog_day_afternoon_OPM/{sub}/meg/{sub}_task-movie_run-{run:03d}_channels.tsv"
                df = pd.read_csv(zf.open(name), sep="\t")
                df = df[df["status"] == "good"].copy()
                df["sensor"] = df["name"].str.replace(r"\s*\[[XYZ]\]", "", regex=True)
                df["axis"] = df["name"].str.extract(r"\[([XYZ])\]")
                rows = []
                for sensor in common_sensors:
                    block = df[df["sensor"] == sensor].copy()
                    block["axis"] = pd.Categorical(block["axis"], categories=["X", "Y", "Z"], ordered=True)
                    block = block.sort_values("axis")
                    if len(block) != 3:
                        raise RuntimeError(f"{sub} run {run} sensor {sensor} has {len(block)} axes")
                    rows.append(block[["Px", "Py", "Pz", "Ox", "Oy", "Oz"]].to_numpy().reshape(-1))
                samples.append(
                    {
                        "subject_id": sid,
                        "run_id": run,
                        "feature": np.concatenate(rows).astype(np.float32),
                    }
                )

    X = np.stack([item["feature"] for item in samples], axis=0)
    y = np.asarray([item["subject_id"] for item in samples], dtype=np.int64)
    run_ids = np.asarray([item["run_id"] for item in samples], dtype=np.int64)

    # Position/orientation variability across subjects for the same sensor slot.
    with zipfile.ZipFile(ZIP_PATH) as zf:
        per_subject_rows = []
        for sub in subjects:
            name = f"dog_day_afternoon_OPM/{sub}/meg/{sub}_task-movie_run-001_channels.tsv"
            df = pd.read_csv(zf.open(name), sep="\t")
            df = df[df["status"] == "good"].copy()
            df["sensor"] = df["name"].str.replace(r"\s*\[[XYZ]\]", "", regex=True)
            df["axis"] = df["name"].str.extract(r"\[([XYZ])\]")
            df = df[df["sensor"].isin(common_sensors)].copy()
            per_subject_rows.append(df.assign(subject=sub))
        geom_df = pd.concat(per_subject_rows, ignore_index=True)

    pos_df = geom_df.groupby(["subject", "sensor"], as_index=False).agg({"Px": "mean", "Py": "mean", "Pz": "mean"})
    pos_var = pos_df.groupby("sensor").agg(Px_std=("Px", "std"), Py_std=("Py", "std"), Pz_std=("Pz", "std"))
    pos_var["pos_std_mean"] = pos_var.mean(axis=1)

    ori_var = geom_df.groupby(["sensor", "axis"]).agg(Ox_std=("Ox", "std"), Oy_std=("Oy", "std"), Oz_std=("Oz", "std"))
    ori_var["ori_std_mean"] = ori_var.mean(axis=1)

    subject_pos = []
    for sub in subjects:
        sdf = pos_df[pos_df["subject"] == sub].set_index("sensor").loc[common_sensors]
        subject_pos.append(sdf[["Px", "Py", "Pz"]].to_numpy().reshape(-1))
    subject_pos = np.asarray(subject_pos, dtype=np.float64)

    return {
        "X": X,
        "y": y,
        "run_ids": run_ids,
        "position_std_mean_m": float(pos_var["pos_std_mean"].mean()),
        "position_std_median_m": float(pos_var["pos_std_mean"].median()),
        "position_std_max_m": float(pos_var["pos_std_mean"].max()),
        "orientation_component_std_mean": float(ori_var["ori_std_mean"].mean()),
        "orientation_component_std_median": float(ori_var["ori_std_mean"].median()),
        "orientation_component_std_max": float(ori_var["ori_std_mean"].max()),
        "pairwise_subject_position_distance_mean": float(np.mean(pdist(subject_pos))),
        "pairwise_subject_position_distance_min": float(np.min(pdist(subject_pos))),
    }


def evaluate_feature_family(name: str, fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Dict:
    feat = fn(x)
    acc = cross_run_accuracy(feat, y, run_ids)
    genuine, impostor = verification_scores(feat, y, run_ids)
    return {
        "feature_dim": int(feat.shape[1]),
        "mean_acc": acc["mean_acc"],
        "std_acc": acc["std_acc"],
        "folds": acc["folds"],
        "eer": compute_eer(genuine, impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "description": name,
    }


def main() -> None:
    global CACHE_DIR, ZIP_PATH, OUT_PATH
    parser = argparse.ArgumentParser(description="Sensor-space and geometry confound controls for OPM-MEG subject identification.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("sensor_confound_controls_beta.json"))
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    ZIP_PATH = args.zip_path
    OUT_PATH = args.output_json

    data = load_cached_beta()
    x = data["X"]
    y = data["y"]
    run_ids = data["run_ids"]
    common_sensors = data["meta"]["common_sensors"]

    geom = load_geometry_features(common_sensors)
    geom_acc = cross_run_accuracy(geom["X"], geom["y"], geom["run_ids"])

    dynamic_results = {
        "baseline_logvar": evaluate_feature_family("Baseline tri-axial LogVar", feat_logvar, x, y, run_ids),
        "relative_logvar": evaluate_feature_family("Per-window mean-centered tri-axial LogVar", feat_relative_logvar, x, y, run_ids),
        "sensor_power": evaluate_feature_family("Orientation-invariant per-sensor power", feat_sensor_power, x, y, run_ids),
        "relative_sensor_power": evaluate_feature_family("Per-window mean-centered orientation-invariant sensor power", feat_relative_sensor_power, x, y, run_ids),
    }

    out = {
        "dataset": "dog_day_afternoon_OPM",
        "band": "beta",
        "window_s": 5,
        "n_subjects": 10,
        "n_common_sensors": len(common_sensors),
        "common_sensors": common_sensors,
        "geometry_only_control": {
            "feature_dim": int(geom["X"].shape[1]),
            "mean_acc": geom_acc["mean_acc"],
            "std_acc": geom_acc["std_acc"],
            "folds": geom_acc["folds"],
            "same_subject_run_geometry_identical": True,
            "position_std_mean_m": geom["position_std_mean_m"],
            "position_std_median_m": geom["position_std_median_m"],
            "position_std_max_m": geom["position_std_max_m"],
            "orientation_component_std_mean": geom["orientation_component_std_mean"],
            "orientation_component_std_median": geom["orientation_component_std_median"],
            "orientation_component_std_max": geom["orientation_component_std_max"],
            "pairwise_subject_position_distance_mean": geom["pairwise_subject_position_distance_mean"],
            "pairwise_subject_position_distance_min": geom["pairwise_subject_position_distance_min"],
        },
        "dynamic_feature_controls": dynamic_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
