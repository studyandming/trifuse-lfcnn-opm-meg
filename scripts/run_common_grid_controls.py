import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_utils import default_cache_dir, default_result_path, default_zip_path

CACHE_DIR = default_cache_dir()
ZIP_PATH = default_zip_path()
OUT_PATH = default_result_path("common_grid_controls_beta.json")

AXES = ("X", "Y", "Z")


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


def read_subject_geometry(common_sensors: List[str]) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for subject in range(1, 11):
            for run in (1, 2):
                name = f"dog_day_afternoon_OPM/sub-{subject:03d}/meg/sub-{subject:03d}_task-movie_run-{run:03d}_channels.tsv"
                df = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t")
                df = df[df["status"].astype(str).str.lower() == "good"].copy()
                df["sensor"] = df["name"].str.replace(r"\s*\[[XYZ]\]$", "", regex=True)
                df["axis"] = df["name"].str.extract(r"\[([XYZ])\]")
                pos_rows = []
                basis_rows = []
                for sensor in common_sensors:
                    block = df[df["sensor"] == sensor].copy()
                    block["axis"] = pd.Categorical(block["axis"], categories=list(AXES), ordered=True)
                    block = block.sort_values("axis")
                    if len(block) != 3:
                        raise RuntimeError(f"subject {subject} run {run} sensor {sensor} has {len(block)} axes")
                    pos_rows.append(block[["Px", "Py", "Pz"]].iloc[0].to_numpy(dtype=np.float64))
                    basis_rows.append(block[["Ox", "Oy", "Oz"]].to_numpy(dtype=np.float64).T)
                out[(subject, run)] = {
                    "positions": np.stack(pos_rows, axis=0),
                    "basis": np.stack(basis_rows, axis=0),
                }
    return out


def build_subject_average_grid(subject_geom: Dict[Tuple[int, int], Dict[str, np.ndarray]]) -> np.ndarray:
    per_subject = []
    for subject in range(1, 11):
        geom = subject_geom[(subject, 1)]
        per_subject.append(geom["positions"])
    return np.mean(np.stack(per_subject, axis=0), axis=0)


def idw_weights(src_xyz: np.ndarray, tgt_xyz: np.ndarray, power: float = 2.0, eps: float = 1e-8) -> np.ndarray:
    dists = np.linalg.norm(tgt_xyz[:, None, :] - src_xyz[None, :, :], axis=-1)
    weights = np.zeros_like(dists)
    exact = dists < 1e-9
    for i in range(dists.shape[0]):
        if np.any(exact[i]):
            j = int(np.argmax(exact[i]))
            weights[i, j] = 1.0
        else:
            w = 1.0 / np.power(dists[i] + eps, power)
            weights[i] = w / w.sum()
    return weights.astype(np.float32)


def build_common_grid_features(
    x: np.ndarray,
    subject_ids: np.ndarray,
    run_ids: np.ndarray,
    common_sensors: List[str],
) -> Dict[str, np.ndarray]:
    subject_geom = read_subject_geometry(common_sensors)
    canonical_pos = build_subject_average_grid(subject_geom)

    n_samples = x.shape[0]
    n_sensors = len(common_sensors)

    out_world_grid = np.empty((n_samples, 3, n_sensors), dtype=np.float32)
    out_world_grid_rel = np.empty((n_samples, 3, n_sensors), dtype=np.float32)
    out_power_grid = np.empty((n_samples, n_sensors), dtype=np.float32)
    out_power_grid_rel = np.empty((n_samples, n_sensors), dtype=np.float32)
    displacement_stats = []

    for subject in range(1, 11):
        for run in (1, 2):
            idx = np.where((subject_ids == subject) & (run_ids == run))[0]
            if idx.size == 0:
                continue

            geom = subject_geom[(subject, run)]
            actual_pos = geom["positions"].astype(np.float32)
            actual_basis = geom["basis"].astype(np.float32)
            weights = idw_weights(actual_pos, canonical_pos.astype(np.float32))
            displacement_stats.append(np.linalg.norm(actual_pos - canonical_pos, axis=1))

            x_group = x[idx]

            # Convert each tri-axial sensor from its local basis to a shared world frame.
            x_world = np.einsum("sab,nbst->nast", actual_basis, x_group, optimize=True)
            world_logv = np.log(np.var(x_world, axis=-1) + 1e-6).astype(np.float32)
            grid_world_logv = np.einsum("ts,nas->nat", weights, world_logv, optimize=True).astype(np.float32)
            out_world_grid[idx] = grid_world_logv

            flat_world = grid_world_logv.reshape(grid_world_logv.shape[0], -1)
            flat_world_rel = flat_world - flat_world.mean(axis=1, keepdims=True)
            out_world_grid_rel[idx] = flat_world_rel.reshape(grid_world_logv.shape[0], 3, n_sensors).astype(np.float32)

            sensor_power = np.log(np.var(x_group, axis=-1).sum(axis=1) + 1e-6).astype(np.float32)
            grid_power = np.einsum("ts,ns->nt", weights, sensor_power, optimize=True).astype(np.float32)
            out_power_grid[idx] = grid_power
            out_power_grid_rel[idx] = (grid_power - grid_power.mean(axis=1, keepdims=True)).astype(np.float32)

    disp = np.concatenate(displacement_stats, axis=0)
    return {
        "common_grid_world_logvar": out_world_grid.reshape(n_samples, -1).astype(np.float32),
        "relative_common_grid_world_logvar": out_world_grid_rel.reshape(n_samples, -1).astype(np.float32),
        "common_grid_sensor_power": out_power_grid.astype(np.float32),
        "relative_common_grid_sensor_power": out_power_grid_rel.astype(np.float32),
        "mean_sensor_to_canonical_displacement_m": float(np.mean(disp)),
        "median_sensor_to_canonical_displacement_m": float(np.median(disp)),
        "max_sensor_to_canonical_displacement_m": float(np.max(disp)),
    }


def evaluate(name: str, features: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Dict:
    acc = cross_run_accuracy(features, y, run_ids)
    genuine, impostor = verification_scores(features, y, run_ids)
    return {
        "description": name,
        "feature_dim": int(features.shape[1]),
        "mean_acc": acc["mean_acc"],
        "std_acc": acc["std_acc"],
        "folds": acc["folds"],
        "eer": compute_eer(genuine, impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def main() -> None:
    global CACHE_DIR, ZIP_PATH, OUT_PATH
    parser = argparse.ArgumentParser(description="World-coordinate common-grid controls for OPM-MEG subject identification.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("common_grid_controls_beta.json"))
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    ZIP_PATH = args.zip_path
    OUT_PATH = args.output_json

    data = load_cached_beta()
    common_sensors = list(data["meta"]["common_sensors"])

    common = build_common_grid_features(
        x=data["X"],
        subject_ids=data["subject_ids"],
        run_ids=data["run_ids"],
        common_sensors=common_sensors,
    )

    out = {
        "dataset": "dog_day_afternoon_OPM",
        "band": "beta",
        "window_s": 5,
        "n_subjects": 10,
        "n_common_sensors": len(common_sensors),
        "canonical_grid_definition": "subject-average positions of the 49 common sensors from run-001",
        "orientation_alignment": "local tri-axial sensor data rotated into shared world coordinates using channels.tsv direction cosines",
        "interpolation": "inverse-distance weighting on sensor positions (power=2)",
        "mean_sensor_to_canonical_displacement_m": common["mean_sensor_to_canonical_displacement_m"],
        "median_sensor_to_canonical_displacement_m": common["median_sensor_to_canonical_displacement_m"],
        "max_sensor_to_canonical_displacement_m": common["max_sensor_to_canonical_displacement_m"],
        "results": {
            "common_grid_world_logvar": evaluate(
                "World-aligned, common-grid LogVar",
                common["common_grid_world_logvar"],
                data["y"],
                data["run_ids"],
            ),
            "relative_common_grid_world_logvar": evaluate(
                "World-aligned, common-grid relative LogVar",
                common["relative_common_grid_world_logvar"],
                data["y"],
                data["run_ids"],
            ),
            "common_grid_sensor_power": evaluate(
                "Common-grid sensor-power",
                common["common_grid_sensor_power"],
                data["y"],
                data["run_ids"],
            ),
            "relative_common_grid_sensor_power": evaluate(
                "Common-grid relative sensor-power",
                common["relative_common_grid_sensor_power"],
                data["y"],
                data["run_ids"],
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
