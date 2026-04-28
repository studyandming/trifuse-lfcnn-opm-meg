import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from release_utils import default_cache_dir, default_result_path

CACHE_DIR = default_cache_dir()
OUT_PATH = default_result_path("multiband_confidence_results.json")
DEFAULT_BANDS = ("alpha", "beta", "broad")
DEFAULT_WINDOW_S = 5
DEFAULT_N_SUB = 10


def cache_tag(n_sub: int) -> str:
    return "_".join(f"sub-{idx:03d}" for idx in range(1, n_sub + 1))


def load_logvar_features(band: str, window_s: int = DEFAULT_WINDOW_S, n_sub: int = DEFAULT_N_SUB) -> Dict[str, np.ndarray]:
    tag = cache_tag(n_sub)
    npz_path = CACHE_DIR / f"{tag}_{band}_200hz_{int(window_s)}s.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    run_ids = data["run_ids"].astype(np.int64)
    features = np.log(np.var(x, axis=-1) + 1e-6).reshape(x.shape[0], -1).astype(np.float32)
    return {
        "features": features,
        "y": y,
        "run_ids": run_ids,
    }


def load_multiband_features(
    bands: Sequence[str],
    window_s: int = DEFAULT_WINDOW_S,
    n_sub: int = DEFAULT_N_SUB,
) -> Dict[str, Dict[str, np.ndarray]]:
    band_data = {band: load_logvar_features(band, window_s=window_s, n_sub=n_sub) for band in bands}
    ref_y = band_data[bands[0]]["y"]
    ref_run_ids = band_data[bands[0]]["run_ids"]
    for band in bands[1:]:
        if not np.array_equal(ref_y, band_data[band]["y"]):
            raise RuntimeError(f"label mismatch across bands: {band}")
        if not np.array_equal(ref_run_ids, band_data[band]["run_ids"]):
            raise RuntimeError(f"run-id mismatch across bands: {band}")
    return band_data


def cosine_scores(probe_features: np.ndarray, templates: np.ndarray) -> np.ndarray:
    return 1.0 - cdist(probe_features, templates, metric="cosine")


def midpoint_threshold(enroll_scores: np.ndarray, enroll_labels: np.ndarray, gallery_subjects: np.ndarray) -> float:
    subject_to_col = {int(subject): idx for idx, subject in enumerate(gallery_subjects)}
    true_cols = np.asarray([subject_to_col[int(label)] for label in enroll_labels], dtype=np.int64)
    genuine = enroll_scores[np.arange(len(enroll_labels)), true_cols]
    mask = np.ones_like(enroll_scores, dtype=bool)
    mask[np.arange(len(enroll_labels)), true_cols] = False
    impostor = enroll_scores[mask]
    return float((np.mean(genuine) + np.mean(impostor)) / 2.0)


def rank1_summary_from_scores(score_matrix: np.ndarray, gallery_subjects: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    pred = gallery_subjects[np.argmax(score_matrix, axis=1)]
    acc = float(np.mean(pred == y_true))
    return {
        "acc": acc,
        "coverage": 1.0,
        "accepted_acc": acc,
    }


def majority_vote(predictions: List[int], tie_break: int) -> int:
    uniq, counts = np.unique(predictions, return_counts=True)
    max_count = counts.max()
    winners = uniq[counts == max_count]
    if len(winners) == 1:
        return int(winners[0])
    return int(tie_break)


def build_directional_state(
    band_data: Dict[str, Dict[str, np.ndarray]],
    bands: Sequence[str],
    enroll_run: int,
    probe_run: int,
    gallery_subjects: np.ndarray,
) -> Dict:
    y = band_data[bands[0]]["y"]
    run_ids = band_data[bands[0]]["run_ids"]
    enroll_mask = (run_ids == enroll_run) & np.isin(y, gallery_subjects)
    probe_mask = run_ids == probe_run
    y_true = y[probe_mask]

    per_band = {}
    for band in bands:
        features = band_data[band]["features"]
        templates = np.stack([features[enroll_mask & (y == subject)].mean(axis=0) for subject in gallery_subjects], axis=0)
        enroll_scores = cosine_scores(features[enroll_mask], templates)
        probe_scores = cosine_scores(features[probe_mask], templates)
        order = np.argsort(-probe_scores, axis=1)
        best_cols = order[:, 0]
        best_ids = gallery_subjects[best_cols]
        best_scores = probe_scores[np.arange(len(y_true)), best_cols]
        second_scores = probe_scores[np.arange(len(y_true)), order[:, 1]]
        per_band[band] = {
            "threshold": midpoint_threshold(enroll_scores, y[enroll_mask], gallery_subjects),
            "score_matrix": probe_scores,
            "best_ids": best_ids,
            "best_scores": best_scores,
            "margin_scores": best_scores - second_scores,
        }

    mean_score_matrix = sum(per_band[band]["score_matrix"] for band in bands) / float(len(bands))
    beta_tie_break = per_band["beta"]["best_ids"] if "beta" in per_band else per_band[bands[0]]["best_ids"]
    return {
        "gallery_subjects": gallery_subjects,
        "y_true": y_true,
        "per_band": per_band,
        "mean_score_matrix": mean_score_matrix,
        "beta_tie_break": beta_tie_break,
    }


def evaluate_high_confidence_closed_set(state: Dict, bands: Sequence[str]) -> Dict:
    y_true = state["y_true"]
    gallery_subjects = state["gallery_subjects"]
    per_band = state["per_band"]
    beta_tie_break = state["beta_tie_break"]

    results = {
        "score_fusion_no_reject": rank1_summary_from_scores(state["mean_score_matrix"], gallery_subjects, y_true),
    }

    majority_pred = []
    for idx in range(len(y_true)):
        votes = [int(per_band[band]["best_ids"][idx]) for band in bands]
        majority_pred.append(majority_vote(votes, tie_break=int(beta_tie_break[idx])))
    majority_pred = np.asarray(majority_pred, dtype=np.int64)
    results["majority_no_reject"] = {
        "acc": float(np.mean(majority_pred == y_true)),
        "coverage": 1.0,
        "accepted_acc": float(np.mean(majority_pred == y_true)),
    }

    strict_accept = np.ones(len(y_true), dtype=bool)
    strict_accept_thr = np.ones(len(y_true), dtype=bool)
    for band in bands:
        strict_accept &= per_band[band]["best_ids"] == per_band[bands[0]]["best_ids"]
        strict_accept_thr &= per_band[band]["best_scores"] >= per_band[band]["threshold"]

    strict_accept_only = strict_accept
    strict_accept_bestthr = strict_accept & strict_accept_thr

    def summarize_accept(accept_mask: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        if not np.any(accept_mask):
            return {
                "coverage": 0.0,
                "accepted_acc": 0.0,
                "overall_acc_with_reject_as_wrong": 0.0,
            }
        accepted_correct = predictions[accept_mask] == y_true[accept_mask]
        return {
            "coverage": float(np.mean(accept_mask)),
            "accepted_acc": float(np.mean(accepted_correct)),
            "overall_acc_with_reject_as_wrong": float(np.mean(accept_mask & (predictions == y_true))),
        }

    results["strict_agree_only"] = summarize_accept(strict_accept_only, majority_pred)
    results["strict_agree_bestthr"] = summarize_accept(strict_accept_bestthr, majority_pred)

    majority_accept = np.zeros(len(y_true), dtype=bool)
    majority_confident_pred = np.full(len(y_true), -1, dtype=np.int64)
    for idx in range(len(y_true)):
        valid_votes = []
        for band in bands:
            if per_band[band]["best_scores"][idx] >= per_band[band]["threshold"]:
                valid_votes.append(int(per_band[band]["best_ids"][idx]))
        if len(valid_votes) >= 2:
            uniq, counts = np.unique(valid_votes, return_counts=True)
            max_count = counts.max()
            winners = uniq[counts == max_count]
            if max_count >= 2 and len(winners) == 1:
                majority_accept[idx] = True
                majority_confident_pred[idx] = int(winners[0])

    results["majority_bestthr"] = summarize_accept(majority_accept, majority_confident_pred)
    return results


def evaluate_open_set_directional(
    band_data: Dict[str, Dict[str, np.ndarray]],
    bands: Sequence[str],
    enroll_run: int = 1,
    probe_run: int = 2,
    n_unknown: int = 2,
    n_repeats: int = 20,
) -> Dict:
    y = band_data[bands[0]]["y"]
    subjects = np.unique(y)
    results = {"score_fusion": [], "majority": [], "strict": []}

    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        unknown_subs = np.sort(rng.choice(subjects, size=n_unknown, replace=False))
        known_subs = np.asarray([subject for subject in subjects if subject not in unknown_subs], dtype=np.int64)
        state = build_directional_state(
            band_data=band_data,
            bands=bands,
            enroll_run=enroll_run,
            probe_run=probe_run,
            gallery_subjects=known_subs,
        )
        y_true = state["y_true"]
        is_unknown = np.isin(y_true, unknown_subs)
        known_mask = ~is_unknown
        per_band = state["per_band"]
        beta_tie_break = state["beta_tie_break"]

        mean_score_matrix = state["mean_score_matrix"]
        fusion_pred = known_subs[np.argmax(mean_score_matrix, axis=1)]
        fusion_best = np.max(mean_score_matrix, axis=1)

        # Fusion threshold is calibrated on enrollment data only.
        enroll_scores = []
        run_ids = band_data[bands[0]]["run_ids"]
        enroll_mask = (run_ids == enroll_run) & np.isin(y, known_subs)
        for band in bands:
            features = band_data[band]["features"]
            templates = np.stack([features[enroll_mask & (y == subject)].mean(axis=0) for subject in known_subs], axis=0)
            enroll_scores.append(cosine_scores(features[enroll_mask], templates))
        mean_enroll_scores = sum(enroll_scores) / float(len(bands))
        fusion_threshold = midpoint_threshold(mean_enroll_scores, y[enroll_mask], known_subs)
        fusion_accept = fusion_best >= fusion_threshold

        def summarize(accept_mask: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
            return {
                "known_acc": float(np.mean((pred[known_mask] == y_true[known_mask]) & accept_mask[known_mask])) if np.any(known_mask) else 0.0,
                "unknown_reject": float(np.mean(~accept_mask[is_unknown])) if np.any(is_unknown) else 0.0,
                "known_coverage": float(np.mean(accept_mask[known_mask])) if np.any(known_mask) else 0.0,
            }

        results["score_fusion"].append(summarize(fusion_accept, fusion_pred))

        majority_accept = np.zeros(len(y_true), dtype=bool)
        majority_pred = np.full(len(y_true), -1, dtype=np.int64)
        strict_accept = np.zeros(len(y_true), dtype=bool)
        strict_pred = np.full(len(y_true), -1, dtype=np.int64)
        for idx in range(len(y_true)):
            valid_votes = []
            for band in bands:
                if per_band[band]["best_scores"][idx] >= per_band[band]["threshold"]:
                    valid_votes.append(int(per_band[band]["best_ids"][idx]))

            if len(valid_votes) >= 2:
                uniq, counts = np.unique(valid_votes, return_counts=True)
                max_count = counts.max()
                winners = uniq[counts == max_count]
                if max_count >= 2 and len(winners) == 1:
                    majority_accept[idx] = True
                    majority_pred[idx] = int(winners[0])
            if len(valid_votes) == len(bands) and len(set(valid_votes)) == 1:
                strict_accept[idx] = True
                strict_pred[idx] = int(valid_votes[0])

        results["majority"].append(summarize(majority_accept, majority_pred))
        results["strict"].append(summarize(strict_accept, strict_pred))

    summary = {}
    for policy, items in results.items():
        summary[policy] = {
            "known_acc": float(np.mean([item["known_acc"] for item in items])),
            "unknown_reject": float(np.mean([item["unknown_reject"] for item in items])),
            "known_coverage": float(np.mean([item["known_coverage"] for item in items])),
        }
    return summary


def main() -> None:
    global CACHE_DIR, OUT_PATH
    parser = argparse.ArgumentParser(description="Multiband high-confidence and open-set rejection diagnostics.")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--output-json", type=Path, default=default_result_path("multiband_confidence_results.json"))
    parser.add_argument("--bands", nargs="+", default=list(DEFAULT_BANDS), choices=["alpha", "beta", "broad"])
    args = parser.parse_args()
    CACHE_DIR = args.cache_dir
    OUT_PATH = args.output_json
    bands = tuple(args.bands)
    band_data = load_multiband_features(bands)
    subjects = np.unique(band_data[bands[0]]["y"])

    directional_state = build_directional_state(
        band_data=band_data,
        bands=bands,
        enroll_run=1,
        probe_run=2,
        gallery_subjects=subjects,
    )
    closed_set_results = evaluate_high_confidence_closed_set(directional_state, bands)

    open_set_results = {}
    for n_unknown in (1, 2, 3):
        open_set_results[f"{n_unknown}unk"] = evaluate_open_set_directional(
            band_data=band_data,
            bands=bands,
            enroll_run=1,
            probe_run=2,
            n_unknown=n_unknown,
            n_repeats=20,
        )

    output = {
        "protocol": {
            "bands": list(bands),
            "feature": "logvar",
            "window_s": DEFAULT_WINDOW_S,
            "subjects": int(len(subjects)),
            "directional_enrollment_probe": [1, 2],
            "note": "Closed-set confidence analysis and open-set multiband rejection on the original OPM benchmark.",
        },
        "closed_set_high_confidence": closed_set_results,
        "open_set_multiband": open_set_results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
