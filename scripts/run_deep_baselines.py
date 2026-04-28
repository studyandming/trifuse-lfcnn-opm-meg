import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from prepare_opm_movie_dataset import (
    BAND_MAP,
    make_deep_ready,
    prepare_dataset,
    stratified_train_val_indices,
    subject_token,
)
from release_utils import default_cache_dir, default_result_path, default_zip_path


class FlattenedWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X is (N, 3, S, T). Classical EEG/BCI baselines see 3*S channels.
        n, axes, sensors, time = X.shape
        flat = X.reshape(n, axes * sensors, time)
        self.X = torch.from_numpy(flat[:, None].astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvClassifierMixin:
    def _infer_flatten_dim(self, n_channels: int, n_times: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.features(dummy)
            return int(out.flatten(1).shape[1])


class EEGNet(nn.Module, ConvClassifierMixin):
    """Compact EEGNet baseline adapted to flattened tri-axial OPM channels."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 125,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._infer_flatten_dim(n_channels, n_times), n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module, ConvClassifierMixin):
    """Shallow ConvNet/FBCSP-style baseline for EEG decoding."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_size: int = 75,
        pool_stride: int = 15,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, temporal_kernel), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            Square(),
            nn.AvgPool2d((1, pool_size), stride=(1, pool_stride)),
            SafeLog(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._infer_flatten_dim(n_channels, n_times), n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class DeepConvNet(nn.Module, ConvClassifierMixin):
    """Deep ConvNet baseline from classical EEG decoding practice."""

    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, (n_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(25, 50, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(50, 100, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2)),
            nn.Dropout(dropout),
            nn.Conv2d(100, 200, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._infer_flatten_dim(n_channels, n_times), n_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))


class EEGConformer(nn.Module):
    """Compact EEG Conformer baseline adapted to flattened tri-axial OPM channels."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        emb_size: int = 40,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, emb_size, (1, 25), padding=(0, 12), bias=False),
            nn.Conv2d(emb_size, emb_size, (n_channels, 1), groups=emb_size, bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            tokens = self._to_tokens(self.patch_embedding(dummy))
            token_count = int(tokens.shape[1])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size * token_count),
            nn.Linear(emb_size * token_count, n_classes),
        )

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # (B, E, 1, P) -> (B, P, E)
        return x.squeeze(2).permute(0, 2, 1)

    def forward(self, x):
        x = self._to_tokens(self.patch_embedding(x))
        x = self.transformer(x)
        return self.classifier(x.flatten(1))


MODEL_FACTORIES = {
    "eegnet": EEGNet,
    "eegconformer": EEGConformer,
    "shallowconvnet": ShallowConvNet,
    "deepconvnet": DeepConvNet,
}


def make_model(name: str, n_channels: int, n_times: int, n_classes: int) -> nn.Module:
    if name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_FACTORIES[name](n_channels=n_channels, n_times=n_times, n_classes=n_classes)


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
    losses: List[float] = []
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        pred = logits.argmax(dim=1)
        losses.append(float(loss.item()))
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": float(correct / total) if total else 0.0,
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_model_cv(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    run_ids: np.ndarray,
    seed: int,
    device: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> Dict:
    folds: List[Tuple[int, int]] = [(1, 2), (2, 1)]
    fold_items: List[Dict] = []
    n_classes = int(len(np.unique(y)))
    n_channels = int(X.shape[1] * X.shape[2])
    n_times = int(X.shape[3])

    for fold_idx, (train_run, test_run) in enumerate(folds, start=1):
        seed_everything(seed + fold_idx)
        trval_idx = np.where(run_ids == train_run)[0]
        test_idx = np.where(run_ids == test_run)[0]
        tr_rel, va_rel = stratified_train_val_indices(y[trval_idx], val_ratio=0.2, seed=seed + fold_idx)
        tr_idx = trval_idx[tr_rel]
        va_idx = trval_idx[va_rel]

        train_loader = DataLoader(
            FlattenedWindowDataset(X[tr_idx], y[tr_idx]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            FlattenedWindowDataset(X[va_idx], y[va_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            FlattenedWindowDataset(X[test_idx], y[test_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = make_model(model_name, n_channels=n_channels, n_times=n_times, n_classes=n_classes).to(device)
        param_count = count_parameters(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_state = None
        best_val_acc = -1.0
        best_epoch = 0
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            val_stats = eval_epoch(model, val_loader, device)
            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
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
                "sizes": {
                    "train": int(len(tr_idx)),
                    "val": int(len(va_idx)),
                    "test": int(len(test_idx)),
                },
            }
        )
        print(
            f"[{model_name}] fold {train_run}->{test_run}: "
            f"val={val_stats['acc'] * 100:.2f}% test={test_stats['acc'] * 100:.2f}% "
            f"best_epoch={best_epoch}"
        )

    scores = [item["test_acc"] for item in fold_items]
    return {
        "model": model_name,
        "folds": fold_items,
        "mean_test_acc": float(np.mean(scores)),
        "std_test_acc": float(np.std(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Appendix-only deep baselines under the same IJCB OPM-MEG cross-run protocol."
    )
    parser.add_argument("--zip-path", type=Path, default=default_zip_path())
    parser.add_argument("--subjects", nargs="+", default=[str(i) for i in range(1, 11)])
    parser.add_argument("--band", choices=sorted(BAND_MAP), default="beta")
    parser.add_argument("--target-fs", type=int, default=200)
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--movie-s", type=float, default=600.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_FACTORIES),
        default=["eegnet", "shallowconvnet", "deepconvnet"],
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=default_result_path("deep_baseline_benchmark_beta.json"),
    )
    parser.add_argument("--force-rebuild", action="store_true")
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
        force_rebuild=args.force_rebuild,
    )
    X = make_deep_ready(prepared.X)

    print("Prepared dataset")
    print("  subjects:", [subject_token(s) for s in subjects])
    print("  band:", prepared.band_name, prepared.band_hz)
    print("  input:", f"{X.shape[1] * X.shape[2]} channels x {X.shape[3]} samples")
    print("  windows:", int(len(prepared.y)))
    print("  device:", device)
    print("  protocol: train on run-1, test on run-2; then swap; mean over both directions")

    results = {
        "task": "appendix deep baselines for cross-run subject identification",
        "protocol": {
            "closed_set": "train on one same-session run, validate on held-out windows from that training run, test on the opposite run, then swap and average",
            "splits": ["run-1 -> run-2", "run-2 -> run-1"],
            "validation": "20% stratified split from the training run only",
        },
        "subjects": [subject_token(s) for s in subjects],
        "n_subjects": int(len(subjects)),
        "band_name": prepared.band_name,
        "band_hz": [float(prepared.band_hz[0]), float(prepared.band_hz[1])],
        "target_fs": int(prepared.target_fs),
        "window_s": float(prepared.window_s),
        "crop_start_s": float(prepared.crop_start_s),
        "crop_duration_s": float(prepared.crop_duration_s),
        "n_common_triaxial_sensors": int(prepared.n_sensors),
        "n_flattened_channels": int(X.shape[1] * X.shape[2]),
        "n_timepoints": int(X.shape[3]),
        "window_counts": prepared.window_counts,
        "training": {
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": device,
        },
        "results": {},
    }

    for model_name in args.models:
        stats = run_model_cv(
            model_name=model_name,
            X=X,
            y=prepared.y,
            run_ids=prepared.run_ids,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        results["results"][model_name] = stats
        print(
            f"[{model_name}] mean={stats['mean_test_acc'] * 100:.2f}% "
            f"std={stats['std_test_acc'] * 100:.2f}%"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved:", args.output_json)


if __name__ == "__main__":
    main()
