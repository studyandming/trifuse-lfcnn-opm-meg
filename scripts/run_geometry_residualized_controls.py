import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from release_utils import default_cache_dir, default_result_path, default_zip_path
from run_common_grid_controls import build_common_grid_features, load_cached_beta
from run_sensor_confound_controls import feat_logvar, load_geometry_features


OUT_PATH = default_result_path("geometry_residualized_controls_beta.json")


def expand_geometry_per_window(subject_ids: np.ndarray, run_ids: np.ndarray, common_sensors) -> np.ndarray:
    geom_run = load_geometry_features(list(common_sensors))
    geom_map = {
        (int(sid), int(run)): feat.astype(np.float32)
        for feat, sid, run in zip(geom_run["X"], geom_run["y"], geom_run["run_ids"])
    }
    return np.stack(
        [geom_map[(int(sid), int(run))] for sid, run in zip(subject_ids, run_ids)],
        axis=0,
    ).astype(np.float32)


def fit_geometry_residualizer(geom_train: np.ndarray, feat_train: np.ndarray) -> Ridge:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, fit_intercept=True),
    )
    model.fit(geom_train, feat_train)
    return model


def apply_residualizer(model, geom: np.ndarray, feat: np.ndarray) -> np.ndarray:
    return (feat - model.predict(geom)).astype(np.float32)


def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def directional_verification_scores(
    train_feat: np.ndarray,
    test_feat: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(train_y)
    templates = {int(s): train_feat[train_y == s].mean(axis=0) for s in subjects}

    genuine = []
    impostor = []
    for feat, true_id in zip(test_feat, test_y):
        true_id = int(true_id)
        for s in subjects:
            score = 1.0 - cosine(feat, templates[int(s)])
            if int(s) == true_id:
                genuine.append(score)
            else:
                impostor.append(score)
    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def evaluate_feature_family(
    description: str,
    features: np.ndarray,
    geometry: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    residualize: bool,
) -> Dict:
    folds = []
    fold_details = []

    for train_run, test_run in ((1, 2), (2, 1)):
        tr = run_ids == train_run
        te = run_ids == test_run

        if residualize:
            residualizer = fit_geometry_residualizer(geometry[tr], features[tr])
            feat_tr = apply_residualizer(residualizer, geometry[tr], features[tr])
            feat_te = apply_residualizer(residualizer, geometry[te], features[te])
        else:
            feat_tr = features[tr].astype(np.float32)
            feat_te = features[te].astype(np.float32)

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(feat_tr, y[tr])
        pred = clf.predict(feat_te)
        acc = float(accuracy_score(y[te], pred))
        folds.append(acc)
        fold_details.append(
            {
                "train_run": int(train_run),
                "test_run": int(test_run),
                "acc": acc,
            }
        )

    tr = run_ids == 1
    te = run_ids == 2
    if residualize:
        residualizer = fit_geometry_residualizer(geometry[tr], features[tr])
        feat_tr = apply_residualizer(residualizer, geometry[tr], features[tr])
        feat_te = apply_residualizer(residualizer, geometry[te], features[te])
    else:
        feat_tr = features[tr].astype(np.float32)
        feat_te = features[te].astype(np.float32)

    genuine, impostor = directional_verification_scores(feat_tr, feat_te, y[tr], y[te])

    return {
        "description": description,
        "feature_dim": int(features.shape[1]),
        "geometry_feature_dim": int(geometry.shape[1]),
        "train_only_residualization": bool(residualize),
        "folds": folds,
        "fold_details": fold_details,
        "mean_acc": float(np.mean(folds)),
        "std_acc": float(np.std(folds)),
        "eer": compute_eer(genuine, impostor),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def summarize_drop(raw: Dict, residualized: Dict) -> Dict:
    return {
        "mean_acc_drop_pp": float(100.0 * (raw["mean_acc"] - residualized["mean_acc"])),
        "eer_change_pp": float(100.0 * (residualized["eer"] - raw["eer"])),
    }


def main() -> None:
    global OUT_PATH
    import run_common_grid_controls as common_controls
    import run_sensor_confound_controls as sensor_controls

    parser = argparse.ArgumentParser(description="Geometry-residualized controls for OPM-MEG subject identification.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("geometry_residualized_controls_beta.json"))
    args = parser.parse_args()
    common_controls.CACHE_DIR = args.cache_dir
    common_controls.ZIP_PATH = args.zip_path
    sensor_controls.CACHE_DIR = args.cache_dir
    sensor_controls.ZIP_PATH = args.zip_path
    OUT_PATH = args.output_json

    data = load_cached_beta()
    common_sensors = list(data["meta"]["common_sensors"])

    native_logvar = feat_logvar(data["X"])
    common_grid = build_common_grid_features(
        x=data["X"],
        subject_ids=data["subject_ids"],
        run_ids=data["run_ids"],
        common_sensors=common_sensors,
    )["common_grid_world_logvar"]
    geometry = expand_geometry_per_window(data["subject_ids"], data["run_ids"], common_sensors)

    native_raw = evaluate_feature_family(
        "Native sensor-space LogVar",
        native_logvar,
        geometry,
        data["y"],
        data["run_ids"],
        residualize=False,
    )
    native_resid = evaluate_feature_family(
        "Native sensor-space LogVar residualized by channels.tsv geometry",
        native_logvar,
        geometry,
        data["y"],
        data["run_ids"],
        residualize=True,
    )
    common_raw = evaluate_feature_family(
        "World-aligned common-grid LogVar",
        common_grid,
        geometry,
        data["y"],
        data["run_ids"],
        residualize=False,
    )
    common_resid = evaluate_feature_family(
        "World-aligned common-grid LogVar residualized by channels.tsv geometry",
        common_grid,
        geometry,
        data["y"],
        data["run_ids"],
        residualize=True,
    )

    out = {
        "dataset": "dog_day_afternoon_OPM",
        "band": "beta",
        "window_s": 5,
        "n_subjects": 10,
        "n_common_sensors": len(common_sensors),
        "geometry_covariates": "channels.tsv metadata for the 49 common sensors: Px, Py, Pz, Ox, Oy, Oz for each retained tri-axial sensor",
        "residualization": "Training-run Ridge regression predicts dynamic feature from geometry metadata; residual feature = dynamic feature - predicted geometry component",
        "results": {
            "native_logvar_raw": native_raw,
            "native_logvar_residualized": native_resid,
            "native_logvar_drop": summarize_drop(native_raw, native_resid),
            "common_grid_logvar_raw": common_raw,
            "common_grid_logvar_residualized": common_resid,
            "common_grid_logvar_drop": summarize_drop(common_raw, common_resid),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
