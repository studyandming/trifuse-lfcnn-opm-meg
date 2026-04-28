import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare_opm_movie_dataset import (
    BAND_MAP,
    prepare_dataset,
    stratified_train_val_indices,
    subject_token,
    zscore_trial_channel,
)
from release_utils import default_cache_dir, default_result_path, default_zip_path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def per_window_zscore(X: np.ndarray) -> np.ndarray:
    n, axes, sensors, time = X.shape
    flat = X.reshape(n * axes, sensors, time)
    flat = zscore_trial_channel(flat)
    return flat.reshape(n, axes, sensors, time).astype(np.float32, copy=False)


def train_run_standardize(X: np.ndarray, train_idx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # Fit mean/std on training-run windows only and apply to train/val/test.
    mean = X[train_idx].mean(axis=(0, 3), keepdims=True)
    std = X[train_idx].std(axis=(0, 3), keepdims=True)
    return ((X - mean) / (std + eps)).astype(np.float32, copy=False)


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LFBranch(nn.Module):
    def __init__(self, in_ch: int, k_components: int, n_filters: int, kernel_size: int, dropout: float):
        super().__init__()
        self.spatial = nn.Conv1d(in_ch, k_components, kernel_size=1, bias=False)
        self.temporal = nn.Conv1d(
            k_components,
            k_components * n_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=k_components,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(k_components * n_filters)
        self.drop = nn.Dropout(dropout)
        self.out_dim = k_components * n_filters

    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        x = F.elu(self.bn(x))
        x = x.mean(dim=-1)
        return self.drop(x)


class TriFuseLFCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        k_components: int = 8,
        n_filters: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.25,
        hidden: int = 64,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [LFBranch(in_ch, k_components, n_filters, kernel_size, dropout) for _ in range(3)]
        )
        feat_dim = self.branches[0].out_dim
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim * 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        feats = [branch(x[:, axis]) for axis, branch in enumerate(self.branches)]
        stacked = torch.stack(feats, dim=1)
        gate_logits = self.gate(torch.cat(feats, dim=-1)).view(x.size(0), 3, -1)
        weights = torch.softmax(gate_logits, dim=1)
        fused = (weights * stacked).sum(dim=1)
        return self.classifier(fused)


class ConcatLFNet(nn.Module):
    """Tri-axis LF-CNN ablation with identical branches but no gating."""

    def __init__(
        self,
        in_ch: int,
        n_classes: int,
        k_components: int = 8,
        n_filters: int = 8,
        kernel_size: int = 31,
        dropout: float = 0.25,
        hidden: int = 64,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [LFBranch(in_ch, k_components, n_filters, kernel_size, dropout) for _ in range(3)]
        )
        feat_dim = self.branches[0].out_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 3, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        feats = [branch(x[:, axis]) for axis, branch in enumerate(self.branches)]
        return self.classifier(torch.cat(feats, dim=-1))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    norm: str
    k_components: int = 8
    n_filters: int = 8
    kernel_size: int = 31
    dropout: float = 0.25
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 3e-4
    batch_size: int = 32


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_model(config: ModelConfig, in_ch: int, n_classes: int) -> nn.Module:
    if config.model_type == "trifuse":
        return TriFuseLFCNN(
            in_ch=in_ch,
            n_classes=n_classes,
            k_components=config.k_components,
            n_filters=config.n_filters,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            hidden=config.hidden,
        )
    if config.model_type == "concat":
        return ConcatLFNet(
            in_ch=in_ch,
            n_classes=n_classes,
            k_components=config.k_components,
            n_filters=config.n_filters,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            hidden=config.hidden,
        )
    raise ValueError(f"unknown model_type: {config.model_type}")


def normalize_for_fold(X: np.ndarray, train_idx: np.ndarray, norm: str) -> np.ndarray:
    if norm == "window_zscore":
        return per_window_zscore(X)
    if norm == "train_standardize":
        return train_run_standardize(X, train_idx)
    if norm == "none":
        return X.astype(np.float32, copy=False)
    raise ValueError(f"unknown norm: {norm}")


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    losses: List[float] = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        losses.append(float(F.cross_entropy(logits, yb).item()))
        correct += int((logits.argmax(dim=1) == yb).sum().item())
        total += int(yb.numel())
    return {"loss": float(np.mean(losses)) if losses else 0.0, "acc": float(correct / total)}


def run_config(
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    config: ModelConfig,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
) -> Dict:
    folds = [(1, 2), (2, 1)]
    fold_items: List[Dict] = []
    n_classes = int(len(np.unique(y)))
    in_ch = int(X.shape[2])

    for fold_idx, (train_run, test_run) in enumerate(folds, start=1):
        seed_everything(seed + 1000 * fold_idx)
        trval_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]
        tr_rel, va_rel = stratified_train_val_indices(y[trval_idx], val_ratio=0.2, seed=seed + fold_idx)
        tr_idx = trval_idx[tr_rel]
        va_idx = trval_idx[va_rel]

        X_fold = normalize_for_fold(X, tr_idx, config.norm)
        train_loader = DataLoader(
            WindowDataset(X_fold[tr_idx], y[tr_idx]),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            WindowDataset(X_fold[va_idx], y[va_idx]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            WindowDataset(X_fold[test_idx], y[test_idx]),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = make_model(config, in_ch=in_ch, n_classes=n_classes).to(device)
        param_count = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        best_state = None
        best_val = -1.0
        best_epoch = 0
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            val_stats = eval_epoch(model, val_loader, device)
            if val_stats["acc"] > best_val:
                best_val = val_stats["acc"]
                best_epoch = epoch
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
                "best_epoch": int(best_epoch),
                "val_acc": float(val_stats["acc"]),
                "test_acc": float(test_stats["acc"]),
                "parameter_count": int(param_count),
                "sizes": {"train": int(len(tr_idx)), "val": int(len(va_idx)), "test": int(len(test_idx))},
            }
        )
        print(
            f"[{config.name}] {train_run}->{test_run}: "
            f"val={val_stats['acc']*100:.2f}% test={test_stats['acc']*100:.2f}% "
            f"epoch={best_epoch} params={param_count}"
        )

    scores = [item["test_acc"] for item in fold_items]
    vals = [item["val_acc"] for item in fold_items]
    return {
        "config": asdict(config),
        "folds": fold_items,
        "mean_val_acc": float(np.mean(vals)),
        "mean_test_acc": float(np.mean(scores)),
        "std_test_acc": float(np.std(scores)),
    }


def default_configs() -> List[ModelConfig]:
    # These are the two configurations reported in the manuscript:
    # the proposed tri-axis gated LF-CNN and its no-gate concat ablation.
    return [
        ModelConfig(
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
        ),
        ModelConfig(
            name="lfcnn_concat_ablation",
            model_type="concat",
            norm="window_zscore",
            k_components=16,
            n_filters=16,
            kernel_size=31,
            dropout=0.10,
            hidden=256,
            lr=1e-3,
            weight_decay=3e-4,
            batch_size=32,
        ),
    ]


def select_configs(names: Iterable[str]) -> List[ModelConfig]:
    configs = default_configs()
    if not names:
        return configs
    wanted = set(names)
    selected = [c for c in configs if c.name in wanted]
    missing = sorted(wanted - {c.name for c in selected})
    if missing:
        raise ValueError(f"unknown configs: {missing}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="TriFuse-LFCNN and no-gate LF-CNN ablation under the IJCB cross-run protocol.")
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--subjects", nargs="+", default=[str(i) for i in range(1, 11)])
    parser.add_argument("--band", choices=sorted(BAND_MAP), default="beta")
    parser.add_argument("--target-fs", type=int, default=200)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--movie-s", type=float, default=600.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_result_path("trifuse_lfcnn_beta.json"),
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = [int(s) for s in args.subjects]

    prepared = prepare_dataset(
        zip_path=args.zip_path,
        subject_ids=subjects,
        runs=[1, 2],
        band_name=args.band,
        target_fs=args.target_fs,
        window_s=args.window_s,
        movie_s=args.movie_s,
        cache_dir=args.cache_dir,
        force_rebuild=False,
    )
    X = prepared.X.astype(np.float32, copy=False)
    y = prepared.y
    run_ids = prepared.run_ids

    print("Prepared dataset")
    print("  subjects:", [subject_token(s) for s in subjects])
    print("  band:", prepared.band_name, prepared.band_hz)
    print("  shape:", tuple(X.shape))
    print("  windows:", int(len(y)))
    print("  device:", device)
    print("  protocol: validation split from training run only; test run untouched")

    results = {
        "task": "TriFuse-LFCNN and no-gate LF-CNN ablation",
        "protocol": {
            "closed_set": "train on one run, validate on held-out windows from that training run, test on the opposite run, then swap and average",
            "splits": ["run-1 -> run-2", "run-2 -> run-1"],
            "selection": "configs are ranked by mean validation accuracy; test accuracy is reported after selection diagnostics",
        },
        "subjects": [subject_token(s) for s in subjects],
        "n_subjects": int(len(subjects)),
        "band_name": prepared.band_name,
        "band_hz": [float(prepared.band_hz[0]), float(prepared.band_hz[1])],
        "target_fs": int(prepared.target_fs),
        "window_s": float(prepared.window_s),
        "n_common_triaxial_sensors": int(prepared.n_sensors),
        "training": {"epochs": int(args.epochs), "patience": int(args.patience), "seed": int(args.seed), "device": device},
        "configs": {},
    }

    for config in select_configs(args.configs):
        stats = run_config(
            X=X,
            y=y,
            run_ids=run_ids,
            config=config,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
        )
        results["configs"][config.name] = stats
        print(
            f"[{config.name}] mean_val={stats['mean_val_acc']*100:.2f}% "
            f"mean_test={stats['mean_test_acc']*100:.2f}% std={stats['std_test_acc']*100:.2f}%"
        )

    ranked_by_val = sorted(results["configs"].items(), key=lambda kv: kv[1]["mean_val_acc"], reverse=True)
    ranked_by_test = sorted(results["configs"].items(), key=lambda kv: kv[1]["mean_test_acc"], reverse=True)
    results["best_by_validation"] = ranked_by_val[0][0] if ranked_by_val else None
    results["best_by_test_diagnostic"] = ranked_by_test[0][0] if ranked_by_test else None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output_json)
    print("Best by validation:", results["best_by_validation"])
    print("Best by test diagnostic:", results["best_by_test_diagnostic"])


if __name__ == "__main__":
    main()
