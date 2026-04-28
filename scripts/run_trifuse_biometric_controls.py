"""TriFuse-LFCNN biometric and geometry-control diagnostics.

This script keeps the LogVar diagnostics untouched and writes independent
TriFuse-LFCNN result files. It uses the fused representation before the final
classifier as the biometric embedding for verification, open-set, and
geometry-residualized analyses.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import run_common_grid_controls as common_grid_controls
import run_sensor_confound_controls as sensor_controls
from prepare_opm_movie_dataset import BAND_MAP, prepare_dataset, subject_token
from release_utils import default_cache_dir, default_result_path, default_zip_path
from run_common_grid_controls import (
    build_subject_average_grid,
    idw_weights,
    read_subject_geometry,
)
from run_sensor_confound_controls import load_geometry_features
from run_trifuse_lfcnn import (
    ModelConfig,
    WindowDataset,
    count_parameters,
    make_model,
    normalize_for_fold,
    seed_everything,
    stratified_train_val_indices,
    train_epoch,
)


DEFAULT_CONFIG = ModelConfig(
    name="trifuse_lfcnn",
    model_type="trifuse",
    norm="window_zscore",
    k_components=16,
    n_filters=16,
    kernel_size=31,
    dropout=0.10,
    hidden=256,
    lr=1e-3,
    weight_decay=3e-4,
    batch_size=32,
)


FPIR_TARGETS = (0.01, 0.05, 0.10)


def json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return obj


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    print(f"Wrote: {path}")


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.linalg.norm(a)) == 0.0 or float(np.linalg.norm(b)) == 0.0:
        return 0.0
    value = 1.0 - cosine(a, b)
    if math.isnan(value):
        return 0.0
    return float(value)


def compute_eer_from_scores(genuine: np.ndarray, impostor: np.ndarray) -> Dict:
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    scores = np.concatenate([genuine, impostor])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return {
        "eer": float((fpr[idx] + fnr[idx]) / 2.0),
        "threshold_at_eer": float(thresholds[idx]),
        "roc": {
            "fpr": fpr.astype(float),
            "tpr": tpr.astype(float),
            "thresholds": thresholds.astype(float),
        },
        "det": {
            "fpr": fpr.astype(float),
            "fnr": fnr.astype(float),
            "thresholds": thresholds.astype(float),
        },
    }


@torch.no_grad()
def extract_fused_embeddings(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fused TriFuse embeddings, logits, and labels."""
    model.eval()
    embeddings: List[np.ndarray] = []
    logits_out: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for xb, yb in loader:
        xb = xb.to(device)
        feats = [branch(xb[:, axis]) for axis, branch in enumerate(model.branches)]
        stacked = torch.stack(feats, dim=1)
        gate_logits = model.gate(torch.cat(feats, dim=-1)).view(xb.size(0), 3, -1)
        weights = torch.softmax(gate_logits, dim=1)
        fused = (weights * stacked).sum(dim=1)
        logits = model.classifier(fused)
        embeddings.append(fused.detach().cpu().numpy().astype(np.float32))
        logits_out.append(logits.detach().cpu().numpy().astype(np.float32))
        labels.append(yb.numpy().astype(np.int64))

    return np.concatenate(embeddings, axis=0), np.concatenate(logits_out, axis=0), np.concatenate(labels, axis=0)


def run_trifuse_folds(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    config: ModelConfig,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    label: str,
) -> Dict:
    folds = []
    n_classes = int(len(np.unique(y)))
    in_ch = int(X.shape[2])

    for fold_idx, (train_run, test_run) in enumerate(((1, 2), (2, 1)), start=1):
        seed_everything(seed + 1000 * fold_idx)
        trval_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]
        tr_rel, va_rel = stratified_train_val_indices(y[trval_idx], val_ratio=0.2, seed=seed + fold_idx)
        tr_idx = trval_idx[tr_rel]
        va_idx = trval_idx[va_rel]

        X_fold = normalize_for_fold(X, tr_idx, config.norm)
        train_loader = DataLoader(WindowDataset(X_fold[tr_idx], y[tr_idx]), batch_size=config.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(WindowDataset(X_fold[va_idx], y[va_idx]), batch_size=config.batch_size, shuffle=False, num_workers=0)
        trval_loader = DataLoader(WindowDataset(X_fold[trval_idx], y[trval_idx]), batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(WindowDataset(X_fold[test_idx], y[test_idx]), batch_size=config.batch_size, shuffle=False, num_workers=0)

        model = make_model(config, in_ch=in_ch, n_classes=n_classes).to(device)
        param_count = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        best_state = None
        best_epoch = 0
        best_val_acc = -1.0
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb.to(device))
                    pred = logits.argmax(dim=1).cpu().numpy()
                    correct += int(np.sum(pred == yb.numpy()))
                    total += int(yb.numel())
            val_acc = float(correct / total)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        train_embeddings, train_logits, train_labels = extract_fused_embeddings(model, trval_loader, device)
        test_embeddings, test_logits, test_labels = extract_fused_embeddings(model, test_loader, device)
        test_pred = test_logits.argmax(axis=1).astype(np.int64)
        test_acc = float(accuracy_score(test_labels, test_pred))

        fold = {
            "train_run": int(train_run),
            "test_run": int(test_run),
            "best_epoch": int(best_epoch),
            "val_acc": float(best_val_acc),
            "test_acc": test_acc,
            "parameter_count": int(param_count),
            "sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "train_run_all": int(len(trval_idx)), "test": int(len(test_idx))},
            "train_indices": trval_idx.astype(int),
            "test_indices": test_idx.astype(int),
            "train_labels": train_labels.astype(int),
            "test_labels": test_labels.astype(int),
            "train_embeddings": train_embeddings,
            "test_embeddings": test_embeddings,
            "test_logits": test_logits,
            "test_predictions": test_pred.astype(int),
        }
        folds.append(fold)
        print(f"[{label}] {train_run}->{test_run}: val={best_val_acc*100:.2f}% test={test_acc*100:.2f}% epoch={best_epoch}")

    mean_acc = float(np.mean([fold["test_acc"] for fold in folds]))
    std_acc = float(np.std([fold["test_acc"] for fold in folds]))
    return {
        "config": asdict(config),
        "folds": folds,
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
    }


def prediction_records(fold: Dict) -> List[Dict]:
    logits = fold["test_logits"]
    records = []
    for row, idx in enumerate(fold["test_indices"]):
        records.append(
            {
                "window_index": int(idx),
                "test_run": int(fold["test_run"]),
                "true_label": int(fold["test_labels"][row]),
                "pred_label": int(fold["test_predictions"][row]),
                "logits": logits[row].astype(float),
            }
        )
    return records


def template_scores(train_embeddings: np.ndarray, train_labels: np.ndarray, test_embeddings: np.ndarray, test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    subjects = np.unique(train_labels)
    templates = {int(s): train_embeddings[train_labels == s].mean(axis=0) for s in subjects}
    genuine: List[float] = []
    impostor: List[float] = []
    for feat, true_id in zip(test_embeddings, test_labels):
        for subject in subjects:
            score = safe_cosine(feat, templates[int(subject)])
            if int(subject) == int(true_id):
                genuine.append(score)
            else:
                impostor.append(score)
    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def verification_summary(beta_run: Dict) -> Dict:
    fold = next(item for item in beta_run["folds"] if item["train_run"] == 1 and item["test_run"] == 2)
    genuine, impostor = template_scores(
        fold["train_embeddings"],
        fold["train_labels"],
        fold["test_embeddings"],
        fold["test_labels"],
    )
    summary = compute_eer_from_scores(genuine, impostor)
    summary.update(
        {
            "train_run": 1,
            "test_run": 2,
            "score": "cosine similarity to subject template in TriFuse fused-embedding space",
            "n_genuine": int(len(genuine)),
            "n_impostor": int(len(impostor)),
            "genuine_scores": genuine,
            "impostor_scores": impostor,
            "genuine_mean": float(np.mean(genuine)),
            "impostor_mean": float(np.mean(impostor)),
        }
    )
    return summary


def midpoint_threshold(enroll_scores: np.ndarray, enroll_labels: np.ndarray, subjects: Sequence[int]) -> float:
    templates = {int(s): enroll_scores[enroll_labels == s].mean(axis=0) for s in subjects}
    genuine: List[float] = []
    impostor: List[float] = []
    for feat, label in zip(enroll_scores, enroll_labels):
        for subject in subjects:
            score = safe_cosine(feat, templates[int(subject)])
            if int(subject) == int(label):
                genuine.append(score)
            else:
                impostor.append(score)
    return float((np.mean(genuine) + np.mean(impostor)) / 2.0)


def open_set_single_band(fold: Dict, n_repeats: int = 20, fpir_targets: Sequence[float] = FPIR_TARGETS) -> Dict:
    y_train = fold["train_labels"]
    y_test = fold["test_labels"]
    subjects = np.unique(y_train)
    results = {}

    for n_unknown in (1, 2, 3):
        known_accs = []
        unknown_rejects = []
        dir_values = {target: [] for target in fpir_targets}
        details = []
        for rep in range(n_repeats):
            rng = np.random.default_rng(rep)
            unknown = np.sort(rng.choice(subjects, size=n_unknown, replace=False))
            known = np.asarray([s for s in subjects if s not in set(unknown)], dtype=np.int64)
            templates = {int(s): fold["train_embeddings"][y_train == s].mean(axis=0) for s in known}
            threshold = midpoint_threshold(fold["train_embeddings"][np.isin(y_train, known)], y_train[np.isin(y_train, known)], known)

            known_best_scores = []
            known_correct = []
            unknown_best_scores = []
            accepted_known_correct = 0
            known_total = 0
            unknown_rejected = 0
            unknown_total = 0
            for feat, true_id in zip(fold["test_embeddings"], y_test):
                scores = np.asarray([safe_cosine(feat, templates[int(s)]) for s in known], dtype=np.float64)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_id = int(known[best_idx])
                if int(true_id) in set(unknown):
                    unknown_total += 1
                    unknown_best_scores.append(best_score)
                    if best_score < threshold:
                        unknown_rejected += 1
                else:
                    known_total += 1
                    correct = best_id == int(true_id)
                    known_best_scores.append(best_score)
                    known_correct.append(1 if correct else 0)
                    if best_score >= threshold and correct:
                        accepted_known_correct += 1

            known_acc = float(accepted_known_correct / max(known_total, 1))
            unknown_reject = float(unknown_rejected / max(unknown_total, 1))
            known_accs.append(known_acc)
            unknown_rejects.append(unknown_reject)

            known_best_scores_arr = np.asarray(known_best_scores, dtype=np.float64)
            known_correct_arr = np.asarray(known_correct, dtype=np.int64)
            unknown_best_scores_arr = np.asarray(unknown_best_scores, dtype=np.float64)
            thresholds = np.unique(np.concatenate([known_best_scores_arr, unknown_best_scores_arr]))
            thresholds = np.concatenate(([thresholds.min() - 1e-6], thresholds, [thresholds.max() + 1e-6]))
            fpirs = np.asarray([float(np.mean(unknown_best_scores_arr >= thr)) for thr in thresholds], dtype=np.float64)
            dirs = np.asarray([float(np.mean((known_best_scores_arr >= thr) & (known_correct_arr == 1))) for thr in thresholds], dtype=np.float64)
            for target in fpir_targets:
                valid = fpirs <= target
                dir_values[target].append(float(np.max(dirs[valid])) if np.any(valid) else 0.0)

            details.append(
                {
                    "rep": int(rep),
                    "unknown_labels": unknown.astype(int),
                    "threshold": float(threshold),
                    "known_acc": known_acc,
                    "unknown_reject": unknown_reject,
                }
            )

        results[f"{n_unknown}unk"] = {
            "n_unknown": int(n_unknown),
            "n_repeats": int(n_repeats),
            "known_acc": {"mean": float(np.mean(known_accs)), "std": float(np.std(known_accs))},
            "unknown_reject": {"mean": float(np.mean(unknown_rejects)), "std": float(np.std(unknown_rejects))},
            "dir_at_fpir": {
                f"{int(target * 100)}": {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for target, vals in dir_values.items()
            },
            "details": details,
        }
    return results


def expand_geometry_per_window(subject_ids: np.ndarray, run_ids: np.ndarray, common_sensors: Sequence[str]) -> np.ndarray:
    geom_run = load_geometry_features(list(common_sensors))
    geom_map = {
        (int(sid), int(run)): feat.astype(np.float32)
        for feat, sid, run in zip(geom_run["X"], geom_run["y"], geom_run["run_ids"])
    }
    return np.stack([geom_map[(int(sid), int(run))] for sid, run in zip(subject_ids, run_ids)], axis=0).astype(np.float32)


def fit_residualizer(geometry: np.ndarray, embeddings: np.ndarray):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, fit_intercept=True))
    model.fit(geometry, embeddings)
    return model


def residualized_embedding_controls(beta_run: Dict, y: np.ndarray, subject_ids: np.ndarray, run_ids: np.ndarray, common_sensors: Sequence[str]) -> Dict:
    geometry = expand_geometry_per_window(subject_ids, run_ids, common_sensors)
    fold_results = []

    for fold in beta_run["folds"]:
        tr_idx = fold["train_indices"]
        te_idx = fold["test_indices"]
        raw_tr = fold["train_embeddings"]
        raw_te = fold["test_embeddings"]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        residualizer = fit_residualizer(geometry[tr_idx], raw_tr)
        resid_tr = (raw_tr - residualizer.predict(geometry[tr_idx])).astype(np.float32)
        resid_te = (raw_te - residualizer.predict(geometry[te_idx])).astype(np.float32)

        def lr_acc(feat_tr: np.ndarray, feat_te: np.ndarray) -> float:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
            clf.fit(feat_tr, y_tr)
            return float(accuracy_score(y_te, clf.predict(feat_te)))

        raw_acc = lr_acc(raw_tr, raw_te)
        resid_acc = lr_acc(resid_tr, resid_te)
        raw_g, raw_i = template_scores(raw_tr, y_tr, raw_te, y_te)
        resid_g, resid_i = template_scores(resid_tr, y_tr, resid_te, y_te)
        fold_results.append(
            {
                "train_run": int(fold["train_run"]),
                "test_run": int(fold["test_run"]),
                "raw_embedding_lr_acc": raw_acc,
                "residualized_embedding_lr_acc": resid_acc,
                "raw_embedding_eer": compute_eer_from_scores(raw_g, raw_i)["eer"],
                "residualized_embedding_eer": compute_eer_from_scores(resid_g, resid_i)["eer"],
                "embedding_dim": int(raw_tr.shape[1]),
                "geometry_dim": int(geometry.shape[1]),
            }
        )

    raw_accs = [item["raw_embedding_lr_acc"] for item in fold_results]
    resid_accs = [item["residualized_embedding_lr_acc"] for item in fold_results]
    raw_eers = [item["raw_embedding_eer"] for item in fold_results]
    resid_eers = [item["residualized_embedding_eer"] for item in fold_results]
    return {
        "residualization": "Training-run Ridge regression predicts TriFuse fused embedding from channels.tsv geometry metadata; residual = embedding - predicted geometry component.",
        "folds": fold_results,
        "summary": {
            "raw_embedding_lr_mean_acc": float(np.mean(raw_accs)),
            "residualized_embedding_lr_mean_acc": float(np.mean(resid_accs)),
            "acc_drop_pp": float(100.0 * (np.mean(raw_accs) - np.mean(resid_accs))),
            "raw_embedding_mean_eer": float(np.mean(raw_eers)),
            "residualized_embedding_mean_eer": float(np.mean(resid_eers)),
            "eer_increase_pp": float(100.0 * (np.mean(resid_eers) - np.mean(raw_eers))),
        },
    }


def build_common_grid_timeseries(
    X: np.ndarray,
    subject_ids: np.ndarray,
    run_ids: np.ndarray,
    common_sensors: Sequence[str],
) -> Dict:
    subject_geom = read_subject_geometry(list(common_sensors))
    canonical_pos = build_subject_average_grid(subject_geom).astype(np.float32)
    out = np.empty_like(X, dtype=np.float32)
    displacement_stats = []

    for subject in np.unique(subject_ids):
        for run in np.unique(run_ids):
            idx = np.where((subject_ids == subject) & (run_ids == run))[0]
            if idx.size == 0:
                continue
            geom = subject_geom[(int(subject), int(run))]
            actual_pos = geom["positions"].astype(np.float32)
            actual_basis = geom["basis"].astype(np.float32)
            weights = idw_weights(actual_pos, canonical_pos)
            x_group = X[idx]
            x_world = np.einsum("sab,nbst->nast", actual_basis, x_group, optimize=True)
            out[idx] = np.einsum("qs,nasl->naql", weights, x_world, optimize=True).astype(np.float32)
            displacement_stats.append(np.linalg.norm(actual_pos - canonical_pos, axis=1))

    disp = np.concatenate(displacement_stats, axis=0)
    return {
        "X": out,
        "mean_sensor_to_canonical_displacement_m": float(np.mean(disp)),
        "median_sensor_to_canonical_displacement_m": float(np.median(disp)),
        "max_sensor_to_canonical_displacement_m": float(np.max(disp)),
    }


def summarize_run_for_json(run: Dict, include_predictions: bool = False) -> Dict:
    folds = []
    for fold in run["folds"]:
        item = {
            "train_run": int(fold["train_run"]),
            "test_run": int(fold["test_run"]),
            "best_epoch": int(fold["best_epoch"]),
            "val_acc": float(fold["val_acc"]),
            "test_acc": float(fold["test_acc"]),
            "parameter_count": int(fold["parameter_count"]),
            "sizes": fold["sizes"],
        }
        if include_predictions:
            item["test_window_predictions"] = prediction_records(fold)
        folds.append(item)
    return {
        "config": run["config"],
        "mean_test_acc": float(run["mean_test_acc"]),
        "std_test_acc": float(run["std_test_acc"]),
        "folds": folds,
    }


def strict_multiband_open_set(runs_by_band: Dict[str, Dict], n_repeats: int = 20) -> Dict:
    bands = ["alpha", "beta", "broad"]
    folds = {band: next(item for item in runs_by_band[band]["folds"] if item["train_run"] == 1 and item["test_run"] == 2) for band in bands}
    y_train = folds["beta"]["train_labels"]
    y_test = folds["beta"]["test_labels"]
    subjects = np.unique(y_train)
    results = {}

    for n_unknown in (1, 2, 3):
        known_accs = []
        unknown_rejects = []
        details = []
        for rep in range(n_repeats):
            rng = np.random.default_rng(rep)
            unknown = np.sort(rng.choice(subjects, size=n_unknown, replace=False))
            known = np.asarray([s for s in subjects if s not in set(unknown)], dtype=np.int64)
            per_band = {}
            for band in bands:
                fold = folds[band]
                train_mask = np.isin(fold["train_labels"], known)
                templates = {int(s): fold["train_embeddings"][fold["train_labels"] == s].mean(axis=0) for s in known}
                threshold = midpoint_threshold(fold["train_embeddings"][train_mask], fold["train_labels"][train_mask], known)
                per_band[band] = {"templates": templates, "threshold": threshold}

            known_total = 0
            known_correct = 0
            unknown_total = 0
            unknown_rejected = 0
            for row, true_id in enumerate(y_test):
                accepted_ids = []
                passed_all = True
                for band in bands:
                    fold = folds[band]
                    templates = per_band[band]["templates"]
                    scores = np.asarray([safe_cosine(fold["test_embeddings"][row], templates[int(s)]) for s in known], dtype=np.float64)
                    best_idx = int(np.argmax(scores))
                    best_id = int(known[best_idx])
                    best_score = float(scores[best_idx])
                    if best_score < per_band[band]["threshold"]:
                        passed_all = False
                    accepted_ids.append(best_id)
                strict_accept = bool(passed_all and len(set(accepted_ids)) == 1)
                pred_id = accepted_ids[0] if strict_accept else -1
                if int(true_id) in set(unknown):
                    unknown_total += 1
                    if not strict_accept:
                        unknown_rejected += 1
                else:
                    known_total += 1
                    if strict_accept and pred_id == int(true_id):
                        known_correct += 1
            known_acc = float(known_correct / max(known_total, 1))
            unknown_reject = float(unknown_rejected / max(unknown_total, 1))
            known_accs.append(known_acc)
            unknown_rejects.append(unknown_reject)
            details.append(
                {
                    "rep": int(rep),
                    "unknown_labels": unknown.astype(int),
                    "known_acc": known_acc,
                    "unknown_reject": unknown_reject,
                    "thresholds": {band: float(per_band[band]["threshold"]) for band in bands},
                }
            )
        results[f"{n_unknown}unk"] = {
            "n_unknown": int(n_unknown),
            "n_repeats": int(n_repeats),
            "known_acc": {"mean": float(np.mean(known_accs)), "std": float(np.std(known_accs))},
            "unknown_reject": {"mean": float(np.mean(unknown_rejects)), "std": float(np.std(unknown_rejects))},
            "details": details,
        }
    return results


def load_prepared(args, band: str):
    return prepare_dataset(
        zip_path=args.zip_path,
        subject_ids=args.subjects,
        runs=[1, 2],
        band_name=band,
        target_fs=args.target_fs,
        window_s=args.window_s,
        movie_s=args.movie_s,
        cache_dir=args.cache_dir,
        force_rebuild=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TriFuse-LFCNN biometric and confound-control diagnostics.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-dir", type=Path, default=default_result_path("_marker").parent)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--target-fs", type=int, default=200)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--movie-s", type=float, default=600.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--open-set-repeats", type=int, default=20)
    parser.add_argument("--skip-beta", action="store_true")
    parser.add_argument("--skip-multiband", action="store_true")
    parser.add_argument("--skip-common-grid", action="store_true")
    args = parser.parse_args()

    common_grid_controls.ZIP_PATH = args.zip_path
    common_grid_controls.CACHE_DIR = args.cache_dir
    sensor_controls.ZIP_PATH = args.zip_path
    sensor_controls.CACHE_DIR = args.cache_dir

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "dataset": "dog_day_afternoon_OPM",
        "subjects": [subject_token(s) for s in args.subjects],
        "target_fs": int(args.target_fs),
        "window_s": float(args.window_s),
        "model_score": "TriFuse fused embedding cosine similarity",
        "config": asdict(DEFAULT_CONFIG),
        "seed": int(args.seed),
        "device": device,
    }

    runs_by_band: Dict[str, Dict] = {}

    if not args.skip_beta:
        beta = load_prepared(args, "beta")
        print("Running beta TriFuse diagnostics")
        beta_run = run_trifuse_folds(
            beta.X.astype(np.float32, copy=False),
            beta.y,
            beta.run_ids,
            DEFAULT_CONFIG,
            args.seed,
            device,
            args.epochs,
            args.patience,
            "beta",
        )
        runs_by_band["beta"] = beta_run
        beta_payload = {
            "protocol": {**protocol, "band": "beta", "band_hz": beta.band_hz, "n_common_triaxial_sensors": int(beta.n_sensors), "window_counts": beta.window_counts},
            "closed_set": summarize_run_for_json(beta_run, include_predictions=True),
            "verification": verification_summary(beta_run),
            "open_set": open_set_single_band(next(item for item in beta_run["folds"] if item["train_run"] == 1 and item["test_run"] == 2), n_repeats=args.open_set_repeats),
        }
        write_json(args.output_dir / "trifuse_biometric_controls_beta.json", beta_payload)

        geom_payload = {
            "protocol": {**protocol, "band": "beta", "band_hz": beta.band_hz, "n_common_triaxial_sensors": int(beta.n_sensors)},
            "embedding_residualization": residualized_embedding_controls(beta_run, beta.y, beta.subject_ids, beta.run_ids, beta.common_sensors),
        }
        write_json(args.output_dir / "trifuse_geometry_residualized_embeddings.json", geom_payload)

        if not args.skip_common_grid:
            print("Building beta common-grid time-series and training TriFuse")
            common_grid = build_common_grid_timeseries(beta.X, beta.subject_ids, beta.run_ids, beta.common_sensors)
            common_run = run_trifuse_folds(
                common_grid["X"],
                beta.y,
                beta.run_ids,
                DEFAULT_CONFIG,
                args.seed,
                device,
                args.epochs,
                args.patience,
                "beta_common_grid",
            )
            common_payload = {
                "protocol": {
                    **protocol,
                    "band": "beta",
                    "band_hz": beta.band_hz,
                    "input": "world-coordinate tri-axial time series interpolated to subject-average common sensor grid",
                    "anatomical_source_grid": "not_run; available source-grid proxy is LogVar-level, not a reconstructed tri-axial time series input for TriFuse.",
                    "mean_sensor_to_canonical_displacement_m": common_grid["mean_sensor_to_canonical_displacement_m"],
                    "median_sensor_to_canonical_displacement_m": common_grid["median_sensor_to_canonical_displacement_m"],
                    "max_sensor_to_canonical_displacement_m": common_grid["max_sensor_to_canonical_displacement_m"],
                },
                "closed_set": summarize_run_for_json(common_run, include_predictions=False),
                "verification": verification_summary(common_run),
            }
            write_json(args.output_dir / "trifuse_common_grid_beta.json", common_payload)

    if not args.skip_multiband:
        for band in ("alpha", "beta", "broad"):
            if band in runs_by_band:
                continue
            prepared = load_prepared(args, band)
            print(f"Running {band} TriFuse for multiband strict open-set")
            runs_by_band[band] = run_trifuse_folds(
                prepared.X.astype(np.float32, copy=False),
                prepared.y,
                prepared.run_ids,
                DEFAULT_CONFIG,
                args.seed,
                device,
                args.epochs,
                args.patience,
                band,
            )
        multiband_payload = {
            "protocol": {
                **protocol,
                "bands": ["alpha", "beta", "broad"],
                "rule": "accept only if alpha, beta, and broad all pass their enrollment-side midpoint thresholds and predict the same identity",
            },
            "per_band_closed_set": {band: summarize_run_for_json(run, include_predictions=False) for band, run in runs_by_band.items()},
            "strict_multiband_open_set": strict_multiband_open_set(runs_by_band, n_repeats=args.open_set_repeats),
        }
        write_json(args.output_dir / "trifuse_multiband_open_set.json", multiband_payload)


if __name__ == "__main__":
    main()
