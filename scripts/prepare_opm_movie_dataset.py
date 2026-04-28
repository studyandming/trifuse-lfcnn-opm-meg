import argparse
import gc
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from release_utils import default_cache_dir, default_result_path, default_zip_path


AXES = ("X", "Y", "Z")
BAND_MAP = {
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "broad": (4.0, 30.0),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subject_token(subject: int) -> str:
    return f"sub-{int(subject):03d}"


def movie_entry(subject: int, run: int) -> str:
    s = subject_token(subject)
    return f"dog_day_afternoon_OPM/{s}/meg/{s}_task-movie_run-{int(run):03d}_meg.mat"


def channels_entry(subject: int, run: int) -> str:
    s = subject_token(subject)
    return f"dog_day_afternoon_OPM/{s}/meg/{s}_task-movie_run-{int(run):03d}_channels.tsv"


def art_entry(subject: int, run: int) -> str:
    s = subject_token(subject)
    return f"dog_day_afternoon_OPM/derivatives/cleaning/{s}/{s}_task-movie_run-{int(run):03d}_vis_artfcts.mat"


def read_tsv_from_zip(zf: zipfile.ZipFile, entry: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(entry)), sep="\t")


def load_mat_from_zip(zf: zipfile.ZipFile, entry: str) -> Dict:
    return loadmat(io.BytesIO(zf.read(entry)), squeeze_me=True, struct_as_record=False, simplify_cells=True)


def get_good_triaxial_sensors(
    zf: zipfile.ZipFile, subjects: Sequence[int], runs: Sequence[int]
) -> List[str]:
    sensor_sets: List[set] = []
    for subject in subjects:
        for run in runs:
            df = read_tsv_from_zip(zf, channels_entry(subject, run))
            good = df[df["status"].astype(str).str.lower() == "good"].copy()
            base = good["name"].str.replace(r"\s*\[[XYZ]\]$", "", regex=True)
            counts = base.value_counts()
            tri = set(base[base.map(counts) == 3])
            sensor_sets.append(tri)
    common = set.intersection(*sensor_sets)
    return sorted(common)


def build_channel_order(common_sensors: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    for sensor in common_sensors:
        for axis in AXES:
            ordered.append(f"{sensor} [{axis}]")
    return ordered


def make_sos(fs: int, band_hz: Tuple[float, float]):
    lo, hi = band_hz
    return butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")


def blockwise_filter_resample(
    data_st: np.ndarray,
    fs: int,
    target_fs: int,
    band_hz: Tuple[float, float],
    block_channels: int = 12,
) -> np.ndarray:
    sos = make_sos(fs=fs, band_hz=band_hz)
    parts: List[np.ndarray] = []
    for start in range(0, data_st.shape[1], block_channels):
        stop = min(data_st.shape[1], start + block_channels)
        block = np.asarray(data_st[:, start:stop], dtype=np.float32)
        block = sosfiltfilt(sos, block, axis=0).astype(np.float32, copy=False)
        if fs != target_fs:
            block = resample_poly(block, up=target_fs, down=fs, axis=0).astype(np.float32, copy=False)
        parts.append(block)
    return np.concatenate(parts, axis=1)


def zscore_trial_channel(x_nct: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x_nct.mean(axis=-1, keepdims=True)
    std = x_nct.std(axis=-1, keepdims=True)
    return (x_nct - mean) / (std + eps)


def overlap_any(start: int, end: int, intervals: np.ndarray) -> bool:
    if intervals.size == 0:
        return False
    return bool(np.any((start < intervals[:, 1]) & (end > intervals[:, 0])))


@dataclass
class PreparedDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    run_ids: np.ndarray
    n_sensors: int
    target_fs: int
    window_s: float
    band_name: str
    band_hz: Tuple[float, float]
    common_sensors: List[str]
    crop_start_s: float
    crop_duration_s: float
    window_counts: Dict[str, int]


def prepare_dataset(
    zip_path: Path,
    subject_ids: Sequence[int],
    runs: Sequence[int],
    band_name: str,
    target_fs: int,
    window_s: float,
    movie_s: float,
    cache_dir: Path,
    force_rebuild: bool = False,
) -> PreparedDataset:
    band_hz = BAND_MAP[band_name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    subj_tag = "_".join(subject_token(s) for s in subject_ids)
    cache_npz = cache_dir / f"{subj_tag}_{band_name}_{target_fs}hz_{int(window_s)}s.npz"
    cache_json = cache_dir / f"{subj_tag}_{band_name}_{target_fs}hz_{int(window_s)}s.json"

    if cache_npz.exists() and cache_json.exists() and not force_rebuild:
        data = np.load(cache_npz, allow_pickle=True)
        meta = json.loads(cache_json.read_text(encoding="utf-8"))
        return PreparedDataset(
            X=data["X"].astype(np.float32, copy=False),
            y=data["y"].astype(np.int64, copy=False),
            subject_ids=data["subject_ids"].astype(np.int64, copy=False),
            run_ids=data["run_ids"].astype(np.int64, copy=False),
            n_sensors=int(meta["n_sensors"]),
            target_fs=int(meta["target_fs"]),
            window_s=float(meta["window_s"]),
            band_name=str(meta["band_name"]),
            band_hz=tuple(meta["band_hz"]),
            common_sensors=list(meta["common_sensors"]),
            crop_start_s=float(meta["crop_start_s"]),
            crop_duration_s=float(meta["crop_duration_s"]),
            window_counts={str(k): int(v) for k, v in meta["window_counts"].items()},
        )

    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_subjects: List[np.ndarray] = []
    all_runs: List[np.ndarray] = []
    window_counts: Dict[str, int] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        common_sensors = get_good_triaxial_sensors(zf, subjects=subject_ids, runs=runs)
        ordered_channels = build_channel_order(common_sensors)
        n_sensors = len(common_sensors)

        crop_start_s = None
        crop_duration_s = float(movie_s)

        for label, subject in enumerate(subject_ids):
            for run in runs:
                mat = load_mat_from_zip(zf, movie_entry(subject, run))
                fs = int(np.asarray(mat["fs"]).item())
                raw = mat["data"]
                total_samples = int(raw.shape[0])
                crop_len = int(round(movie_s * fs))
                crop_start = max(0, (total_samples - crop_len) // 2)
                crop_end = crop_start + crop_len
                if crop_start_s is None:
                    crop_start_s = crop_start / fs

                df = read_tsv_from_zip(zf, channels_entry(subject, run))
                name_to_idx = {name: idx for idx, name in enumerate(df["name"].tolist())}
                use_idx = [name_to_idx[name] for name in ordered_channels]
                x = np.asarray(raw[:, use_idx], dtype=np.float32)
                del raw
                del mat
                gc.collect()

                x = x[crop_start:crop_end]
                x = blockwise_filter_resample(x, fs=fs, target_fs=target_fs, band_hz=band_hz)

                art = load_mat_from_zip(zf, art_entry(subject, run))["vis_artfcts"]
                art = np.asarray(art, dtype=np.int64).reshape(-1, 2)
                art = art - int(crop_start)
                art[:, 0] = np.clip(art[:, 0], 0, crop_len)
                art[:, 1] = np.clip(art[:, 1], 0, crop_len)
                art = art[art[:, 1] > art[:, 0]]
                if target_fs != fs:
                    art = np.rint(art * (target_fs / fs)).astype(np.int64)

                samples_per_window = int(round(window_s * target_fs))
                starts = np.arange(0, x.shape[0] - samples_per_window + 1, samples_per_window, dtype=np.int64)
                run_windows: List[np.ndarray] = []
                for start in starts:
                    stop = start + samples_per_window
                    if overlap_any(int(start), int(stop), art):
                        continue
                    seg = x[start:stop]
                    seg = seg.reshape(samples_per_window, n_sensors, 3)
                    seg = np.transpose(seg, (2, 1, 0)).astype(np.float32, copy=False)
                    run_windows.append(seg)
                run_key = f"{subject_token(subject)}_run{int(run)}"
                window_counts[run_key] = len(run_windows)
                if not run_windows:
                    raise RuntimeError(f"No usable windows for {run_key}")
                run_arr = np.stack(run_windows, axis=0)
                all_windows.append(run_arr)
                all_labels.append(np.full(len(run_arr), label, dtype=np.int64))
                all_subjects.append(np.full(len(run_arr), subject, dtype=np.int64))
                all_runs.append(np.full(len(run_arr), run, dtype=np.int64))

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subj = np.concatenate(all_subjects, axis=0)
    run_ids = np.concatenate(all_runs, axis=0)

    np.savez(
        cache_npz,
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        subject_ids=subj.astype(np.int64, copy=False),
        run_ids=run_ids.astype(np.int64, copy=False),
    )
    meta = {
        "zip_path": str(zip_path),
        "subject_ids": [int(s) for s in subject_ids],
        "runs": [int(r) for r in runs],
        "band_name": band_name,
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "target_fs": int(target_fs),
        "window_s": float(window_s),
        "movie_s": float(movie_s),
        "crop_start_s": float(crop_start_s),
        "crop_duration_s": float(crop_duration_s),
        "n_sensors": int(n_sensors),
        "n_windows": int(len(y)),
        "common_sensors": list(common_sensors),
        "window_counts": window_counts,
    }
    cache_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return PreparedDataset(
        X=X,
        y=y,
        subject_ids=subj,
        run_ids=run_ids,
        n_sensors=n_sensors,
        target_fs=target_fs,
        window_s=window_s,
        band_name=band_name,
        band_hz=band_hz,
        common_sensors=common_sensors,
        crop_start_s=float(crop_start_s),
        crop_duration_s=float(crop_duration_s),
        window_counts=window_counts,
    )


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SpatialProjection(nn.Module):
    def __init__(self, in_ch: int, k_components: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels=in_ch, out_channels=k_components, kernel_size=1, bias=False)

    def forward(self, x):
        return self.proj(x)


class TemporalBlock(nn.Module):
    def __init__(self, k_components: int, n_filters: int, kernel_size: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels=k_components,
            out_channels=k_components * n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=k_components,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(k_components * n_filters)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = F.elu(x)
        x = x.mean(dim=-1)
        x = self.drop(x)
        return x


class LFBranch(nn.Module):
    def __init__(self, in_ch: int, k_components: int, n_filters: int, kernel_size: int, dropout: float):
        super().__init__()
        self.spatial = SpatialProjection(in_ch=in_ch, k_components=k_components)
        self.temporal = TemporalBlock(k_components, n_filters, kernel_size, dropout)
        self.out_dim = k_components * n_filters

    def forward(self, x):
        x = self.spatial(x)
        return self.temporal(x)


class TriBranchLFNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        mode: str = "trifuse",
        k_components: int = 8,
        n_filters: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.25,
        hidden: int = 64,
    ):
        super().__init__()
        if mode not in {"x", "y", "z", "concat", "trifuse"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.x_branch = LFBranch(in_ch, k_components, n_filters, kernel_size, dropout)
        self.y_branch = LFBranch(in_ch, k_components, n_filters, kernel_size, dropout)
        self.z_branch = LFBranch(in_ch, k_components, n_filters, kernel_size, dropout)
        feat_dim = self.x_branch.out_dim

        if mode == "trifuse":
            self.gate = nn.Sequential(
                nn.Linear(feat_dim * 3, hidden),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, feat_dim * 3),
            )
            cls_in = feat_dim
        elif mode == "concat":
            self.gate = None
            cls_in = feat_dim * 3
        else:
            self.gate = None
            cls_in = feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(cls_in, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        zx = self.x_branch(x[:, 0])
        zy = self.y_branch(x[:, 1])
        zz = self.z_branch(x[:, 2])

        if self.mode == "x":
            fused = zx
            alpha = torch.full((x.size(0), 3), float("nan"), device=x.device)
        elif self.mode == "y":
            fused = zy
            alpha = torch.full((x.size(0), 3), float("nan"), device=x.device)
        elif self.mode == "z":
            fused = zz
            alpha = torch.full((x.size(0), 3), float("nan"), device=x.device)
        elif self.mode == "concat":
            fused = torch.cat([zx, zy, zz], dim=-1)
            alpha = torch.full((x.size(0), 3), float("nan"), device=x.device)
        else:
            feats = torch.stack([zx, zy, zz], dim=1)
            gate_logits = self.gate(torch.cat([zx, zy, zz], dim=-1)).view(x.size(0), 3, -1)
            weights = torch.softmax(gate_logits, dim=1)
            fused = (weights * feats).sum(dim=1)
            alpha = weights.mean(dim=-1)

        logits = self.classifier(fused)
        return logits, alpha


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    alpha_all: List[torch.Tensor] = []
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, alpha = model(xb)
        loss = F.cross_entropy(logits, yb)
        losses.append(float(loss.item()))
        logits_all.append(logits.cpu())
        y_all.append(yb.cpu())
        alpha_all.append(alpha.cpu())
    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0)
    preds = logits.argmax(dim=1)
    acc = float((preds == y).float().mean().item())
    alpha = torch.cat(alpha_all, dim=0)
    alpha_np = alpha.numpy()
    if np.isnan(alpha_np).all():
        alpha_mean = [float("nan")] * int(alpha_np.shape[1])
    else:
        alpha_mean = np.nanmean(alpha_np, axis=0).tolist()
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "alpha_mean": [float(v) for v in alpha_mean],
    }


def make_deep_ready(X: np.ndarray) -> np.ndarray:
    n, branches, c, t = X.shape
    out = X.reshape(n * branches, c, t)
    out = zscore_trial_channel(out)
    return out.reshape(n, branches, c, t).astype(np.float32, copy=False)


def make_logvar_features(X: np.ndarray) -> np.ndarray:
    feats = np.log(np.var(X, axis=-1) + 1e-6)
    return feats.reshape(feats.shape[0], -1).astype(np.float32, copy=False)


def stratified_train_val_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr, va = next(splitter.split(np.zeros(len(y)), y))
    return tr, va


def run_logistic_cv(features: np.ndarray, y: np.ndarray, run_ids: np.ndarray) -> Dict:
    folds = [(1, 2), (2, 1)]
    c_values = [0.1, 1.0, 10.0]
    fold_items: List[Dict] = []
    for train_run, test_run in folds:
        trval_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]
        tr_rel, va_rel = stratified_train_val_indices(y[trval_idx], val_ratio=0.2, seed=100 + train_run)
        tr_idx = trval_idx[tr_rel]
        va_idx = trval_idx[va_rel]

        best = None
        for c in c_values:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, max_iter=2000, solver="lbfgs"),
            )
            clf.fit(features[tr_idx], y[tr_idx])
            va_acc = accuracy_score(y[va_idx], clf.predict(features[va_idx]))
            if best is None or va_acc > best["val_acc"]:
                best = {"model": clf, "C": c, "val_acc": float(va_acc)}

        test_pred = best["model"].predict(features[test_idx])
        test_acc = accuracy_score(y[test_idx], test_pred)
        fold_items.append(
            {
                "train_run": int(train_run),
                "test_run": int(test_run),
                "C": float(best["C"]),
                "val_acc": float(best["val_acc"]),
                "test_acc": float(test_acc),
                "sizes": {
                    "train": int(len(tr_idx)),
                    "val": int(len(va_idx)),
                    "test": int(len(test_idx)),
                },
            }
        )
    scores = [item["test_acc"] for item in fold_items]
    return {
        "mode": "logvar_lr",
        "folds": fold_items,
        "mean_test_acc": float(np.mean(scores)),
        "std_test_acc": float(np.std(scores)),
    }


def run_deep_cv(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    mode: str,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict:
    folds = [(1, 2), (2, 1)]
    fold_items: List[Dict] = []
    seed_everything(seed)
    n_classes = int(len(np.unique(y)))

    for fold_idx, (train_run, test_run) in enumerate(folds, start=1):
        trval_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]
        tr_rel, va_rel = stratified_train_val_indices(y[trval_idx], val_ratio=0.2, seed=seed + fold_idx)
        tr_idx = trval_idx[tr_rel]
        va_idx = trval_idx[va_rel]

        train_loader = DataLoader(WindowDataset(X[tr_idx], y[tr_idx]), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(WindowDataset(X[va_idx], y[va_idx]), batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(WindowDataset(X[test_idx], y[test_idx]), batch_size=batch_size, shuffle=False, num_workers=0)

        model = TriBranchLFNet(in_ch=X.shape[2], n_classes=n_classes, mode=mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-4)

        best_state = None
        best_val = -1.0
        patience = 12
        bad_epochs = 0
        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, device)
            val_stats = eval_epoch(model, val_loader, device)
            if val_stats["acc"] > best_val:
                best_val = val_stats["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_stats = eval_epoch(model, val_loader, device)
        test_stats = eval_epoch(model, test_loader, device)
        fold_items.append(
            {
                "train_run": int(train_run),
                "test_run": int(test_run),
                "val_acc": float(val_stats["acc"]),
                "test_acc": float(test_stats["acc"]),
                "alpha_mean": test_stats["alpha_mean"],
                "sizes": {
                    "train": int(len(tr_idx)),
                    "val": int(len(va_idx)),
                    "test": int(len(test_idx)),
                },
            }
        )

    scores = [item["test_acc"] for item in fold_items]
    return {
        "mode": mode,
        "folds": fold_items,
        "mean_test_acc": float(np.mean(scores)),
        "std_test_acc": float(np.std(scores)),
    }


def parse_subjects(values: Iterable[str]) -> List[int]:
    return [int(v) for v in values]


def main():
    parser = argparse.ArgumentParser(description="Two-subject OPM movie identification validation.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=default_zip_path(),
    )
    parser.add_argument("--subjects", nargs="+", default=["1", "2"])
    parser.add_argument("--band", choices=sorted(BAND_MAP), default="alpha")
    parser.add_argument("--modes", nargs="+", default=["logvar_lr", "trifuse"])
    parser.add_argument("--target-fs", type=int, default=200)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--movie-s", type=float, default=600.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_result_path("opm_movie_validation_results.json"),
    )
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepared = prepare_dataset(
        zip_path=args.zip_path,
        subject_ids=subjects,
        runs=[1, 2],
        band_name=args.band,
        target_fs=args.target_fs,
        window_s=args.window_s,
        movie_s=args.movie_s,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild,
    )

    print("Prepared dataset:")
    print("  subjects:", [subject_token(v) for v in subjects])
    print("  band:", prepared.band_name, prepared.band_hz)
    print("  common triaxial sensors:", prepared.n_sensors)
    print("  crop start (s):", f"{prepared.crop_start_s:.2f}")
    print("  windows per run:", prepared.window_counts)
    print("  total windows:", len(prepared.y))
    print("  device:", device)

    results = {
        "task": "cross-run subject identification",
        "subjects": [subject_token(v) for v in subjects],
        "n_subjects": int(len(subjects)),
        "band_name": prepared.band_name,
        "band_hz": [float(prepared.band_hz[0]), float(prepared.band_hz[1])],
        "target_fs": int(prepared.target_fs),
        "window_s": float(prepared.window_s),
        "crop_start_s": float(prepared.crop_start_s),
        "crop_duration_s": float(prepared.crop_duration_s),
        "n_common_triaxial_sensors": int(prepared.n_sensors),
        "window_counts": prepared.window_counts,
        "results": {},
    }

    modes = args.modes
    if "logvar_lr" in modes:
        feat = make_logvar_features(prepared.X)
        stats = run_logistic_cv(feat, prepared.y, prepared.run_ids)
        results["results"]["logvar_lr"] = stats
        print(f"[logvar_lr] mean_test_acc={stats['mean_test_acc']:.4f} std={stats['std_test_acc']:.4f}")

    deep_modes = [mode for mode in modes if mode != "logvar_lr"]
    if deep_modes:
        deep_X = make_deep_ready(prepared.X)
        for mode in deep_modes:
            stats = run_deep_cv(
                X=deep_X,
                y=prepared.y,
                run_ids=prepared.run_ids,
                mode=mode,
                seed=args.seed,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            results["results"][mode] = stats
            print(f"[{mode}] mean_test_acc={stats['mean_test_acc']:.4f} std={stats['std_test_acc']:.4f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved results to:", args.output_json)


if __name__ == "__main__":
    main()
