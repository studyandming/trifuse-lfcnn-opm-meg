import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import cosine, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_utils import default_cache_dir, default_result_path, default_zip_path

CACHE_DIR = default_cache_dir()
ZIP_PATH = default_zip_path()
OUT_PATH = default_result_path("anatomical_source_grid_controls_beta.json")

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


def read_subject_sensor_geometry(common_sensors: List[str]) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for subject in range(1, 11):
            for run in (1, 2):
                entry = f"dog_day_afternoon_OPM/sub-{subject:03d}/meg/sub-{subject:03d}_task-movie_run-{run:03d}_channels.tsv"
                df = pd.read_csv(io.BytesIO(zf.read(entry)), sep="\t")
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


def read_subject_source_anchors() -> np.ndarray:
    anchors = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for subject in range(1, 11):
            seg_entry = f"dog_day_afternoon_OPM/derivatives/sourcespace/sub-{subject:03d}/sub-{subject:03d}_segmentedmri.mat"
            vox_entry = f"dog_day_afternoon_OPM/derivatives/sourcespace/sub-{subject:03d}/sub-{subject:03d}_voxlox.mat"
            seg = loadmat(io.BytesIO(zf.read(seg_entry)), squeeze_me=True, struct_as_record=False)["segmentedmri"]
            vox = np.asarray(
                loadmat(io.BytesIO(zf.read(vox_entry)), squeeze_me=True, struct_as_record=False)["voxlox"],
                dtype=np.float64,
            )
            world_mm = (np.asarray(seg.transform, dtype=np.float64) @ np.vstack([vox, np.ones(vox.shape[1])]))[:3].T
            anchors.append(world_mm / 1000.0)
    return np.stack(anchors, axis=0)


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
    parser = argparse.ArgumentParser(description="Anatomy-aware source-anchor proxy controls for OPM-MEG subject identification.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("anatomical_source_grid_controls_beta.json"))
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    ZIP_PATH = args.zip_path
    OUT_PATH = args.output_json

    data = load_cached_beta()
    common_sensors = list(data["meta"]["common_sensors"])
    sensor_geom = read_subject_sensor_geometry(common_sensors)
    source_anchors = read_subject_source_anchors()
    canonical_source_grid = source_anchors.mean(axis=0)

    n_samples = data["X"].shape[0]
    n_nodes = canonical_source_grid.shape[0]

    world_logvar = np.empty((n_samples, 3, n_nodes), dtype=np.float32)
    world_logvar_rel = np.empty((n_samples, 3, n_nodes), dtype=np.float32)
    sensor_power = np.empty((n_samples, n_nodes), dtype=np.float32)
    sensor_power_rel = np.empty((n_samples, n_nodes), dtype=np.float32)
    geometry_feature = np.empty((20, 3 * n_nodes), dtype=np.float32)
    canonical_disp = []

    geom_row = 0
    for subject in range(1, 11):
        subj_source = source_anchors[subject - 1]
        canonical_disp.append(np.linalg.norm(subj_source - canonical_source_grid, axis=1))
        geometry_feature[geom_row] = subj_source.reshape(-1).astype(np.float32)
        geometry_feature[geom_row + 1] = subj_source.reshape(-1).astype(np.float32)
        geom_row += 2

        for run in (1, 2):
            idx = np.where((data["subject_ids"] == subject) & (data["run_ids"] == run))[0]
            if idx.size == 0:
                continue

            geom = sensor_geom[(subject, run)]
            src_xyz = geom["positions"].astype(np.float32)
            basis = geom["basis"].astype(np.float32)
            weights = idw_weights(src_xyz, canonical_source_grid.astype(np.float32))
            x_group = data["X"][idx]

            x_world = np.einsum("sab,nbst->nast", basis, x_group, optimize=True)
            sensor_world_logv = np.log(np.var(x_world, axis=-1) + 1e-6).astype(np.float32)
            node_world_logv = np.einsum("ts,nas->nat", weights, sensor_world_logv, optimize=True).astype(np.float32)
            world_logvar[idx] = node_world_logv

            flat = node_world_logv.reshape(node_world_logv.shape[0], -1)
            flat_rel = flat - flat.mean(axis=1, keepdims=True)
            world_logvar_rel[idx] = flat_rel.reshape(node_world_logv.shape[0], 3, n_nodes).astype(np.float32)

            sensor_pow = np.log(np.var(x_group, axis=-1).sum(axis=1) + 1e-6).astype(np.float32)
            node_pow = np.einsum("ts,ns->nt", weights, sensor_pow, optimize=True).astype(np.float32)
            sensor_power[idx] = node_pow
            sensor_power_rel[idx] = (node_pow - node_pow.mean(axis=1, keepdims=True)).astype(np.float32)

    geom_y = np.repeat(np.arange(10, dtype=np.int64), 2)
    geom_run_ids = np.tile(np.array([1, 2], dtype=np.int64), 10)
    geom_acc = cross_run_accuracy(geometry_feature, geom_y, geom_run_ids)
    disp = np.concatenate(canonical_disp, axis=0)

    topology = np.stack([pdist(points) for points in source_anchors], axis=0)
    corr = np.corrcoef(topology)
    corr_tri = corr[np.triu_indices_from(corr, k=1)]

    out = {
        "dataset": "dog_day_afternoon_OPM",
        "band": "beta",
        "window_s": 5,
        "n_subjects": 10,
        "n_common_sensors": len(common_sensors),
        "n_source_anchors": int(n_nodes),
        "source_anchor_definition": "78 source-space anchor nodes from derivatives/sourcespace/*_voxlox.mat, transformed with segmentedmri.transform",
        "canonical_grid_definition": "subject-average anatomical source-anchor coordinates",
        "interpolation": "inverse-distance weighting from sensor positions to anatomical source anchors (power=2)",
        "orientation_alignment": "local tri-axial sensor data rotated into shared world coordinates using channels.tsv direction cosines",
        "source_anchor_topology_corr_mean": float(corr_tri.mean()),
        "source_anchor_topology_corr_median": float(np.median(corr_tri)),
        "source_anchor_topology_corr_min": float(corr_tri.min()),
        "mean_subject_anchor_displacement_to_canonical_m": float(np.mean(disp)),
        "median_subject_anchor_displacement_to_canonical_m": float(np.median(disp)),
        "max_subject_anchor_displacement_to_canonical_m": float(np.max(disp)),
        "geometry_only_control": {
            "feature_dim": int(geometry_feature.shape[1]),
            "mean_acc": geom_acc["mean_acc"],
            "std_acc": geom_acc["std_acc"],
            "folds": geom_acc["folds"],
            "description": "Subject-native source-anchor coordinates only",
        },
        "results": {
            "anatomical_source_grid_world_logvar": evaluate(
                "Anatomical source-grid world LogVar",
                world_logvar.reshape(n_samples, -1).astype(np.float32),
                data["y"],
                data["run_ids"],
            ),
            "relative_anatomical_source_grid_world_logvar": evaluate(
                "Anatomical source-grid relative world LogVar",
                world_logvar_rel.reshape(n_samples, -1).astype(np.float32),
                data["y"],
                data["run_ids"],
            ),
            "anatomical_source_grid_sensor_power": evaluate(
                "Anatomical source-grid sensor-power",
                sensor_power.astype(np.float32),
                data["y"],
                data["run_ids"],
            ),
            "relative_anatomical_source_grid_sensor_power": evaluate(
                "Anatomical source-grid relative sensor-power",
                sensor_power_rel.astype(np.float32),
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
