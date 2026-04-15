from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.classical_features import ClassicalFeatureSpec, extract_classical_features_from_path
from utils.config import get_config
from utils.path_utils import normalize_csv_relative_path


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Train pure scikit-learn classical fallback model.")
    parser.add_argument("--use-base-csv", type=int, default=1, help="1=include train/valid/test CSV splits, 0=local dataset only")
    parser.add_argument("--max-train", type=int, default=12000)
    parser.add_argument("--max-valid", type=int, default=3000)
    parser.add_argument("--max-test", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-root", type=str, default=str(cfg.image_root))

    parser.add_argument("--extra-train-real-root", type=str, default="Dataset/rvf10k/train/real")
    parser.add_argument("--extra-train-fake-root", type=str, default="Dataset/rvf10k/train/fake")
    parser.add_argument("--extra-valid-real-root", type=str, default="Dataset/rvf10k/valid/real")
    parser.add_argument("--extra-valid-fake-root", type=str, default="Dataset/rvf10k/valid/fake")
    parser.add_argument("--extra-train-per-class", type=int, default=2500)
    parser.add_argument("--extra-valid-per-class", type=int, default=900)

    parser.add_argument("--hard-failure-image", type=str, default="data/hard_failures/fake/f3.jpg")
    parser.add_argument("--hard-failure-train-copies", type=int, default=30)
    parser.add_argument("--hard-failure-valid-copies", type=int, default=12)

    parser.add_argument("--quality-filter", type=int, default=1, help="1=enabled, 0=disabled")
    parser.add_argument("--reuse-features", type=int, default=1, help="1=auto-resume from saved NPZ features when present")
    parser.add_argument("--hard-mine-rounds", type=int, default=2)
    parser.add_argument("--hard-mine-repeat", type=int, default=3)
    parser.add_argument("--fake-sample-weight", type=float, default=1.35)
    parser.add_argument("--local-train-per-class", type=int, default=1800)
    parser.add_argument("--local-valid-per-class", type=int, default=500)
    parser.add_argument("--local-test-per-class", type=int, default=500)
    parser.add_argument("--cache-prefix", type=str, default="")
    parser.add_argument("--external-hard-fake-dir", type=str, default="data/hard_examples/fake")
    parser.add_argument("--external-hard-fake-repeat", type=int, default=4)
    parser.add_argument("--external-hard-real-dir", type=str, default="data/hard_examples/real")
    parser.add_argument("--external-hard-real-repeat", type=int, default=4)
    return parser.parse_args()


def _resolve_image_root(arg_root: str) -> Path:
    # Prefer audited Kaggle root when available.
    audit_path = Path("results/dataset_audit_kaggle140k.json")
    if audit_path.exists():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            root_str = (
                payload.get("splits", {})
                .get("train", {})
                .get("resolved_image_root")
            )
            if root_str:
                p = Path(str(root_str))
                if p.exists():
                    return p
        except Exception:
            pass

    summary_path = Path("results/kaggle_dataset_summary.json")
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            image_root = Path(summary.get("image_root", ""))
            if image_root.exists():
                return image_root
        except Exception:
            pass

    root = Path(arg_root)
    if root.exists():
        return root
    raise FileNotFoundError(f"Could not resolve image root from {arg_root} or summary json.")


def _resolve_optional_dir(path_str: str) -> Path | None:
    p = Path(path_str)
    if p.exists() and p.is_dir():
        return p
    return None


def _load_split(csv_path: Path, image_root: Path, limit: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=seed).reset_index(drop=True)

    abs_paths = []
    for rel in df["path"].astype(str).tolist():
        abs_paths.append(str(image_root / normalize_csv_relative_path(rel)))
    df = df.assign(abs_path=abs_paths, source="kaggle_140k")
    df = df[df["abs_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    return df[["path", "label", "label_str", "abs_path", "source"]]


def _collect_dir_split(root: Path | None, label: int, limit: int, seed: int, source: str) -> pd.DataFrame:
    if root is None or not root.exists():
        return pd.DataFrame(columns=["path", "label", "label_str", "abs_path", "source"])
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not files:
        return pd.DataFrame(columns=["path", "label", "label_str", "abs_path", "source"])
    rng = np.random.default_rng(seed)
    if limit > 0 and len(files) > limit:
        idx = rng.choice(len(files), size=limit, replace=False)
        files = [files[i] for i in idx]

    rows = []
    for p in files:
        rows.append(
            {
                "path": str(p),
                "label": int(label),
                "label_str": "real" if int(label) == 1 else "fake",
                "abs_path": str(p),
                "source": source,
            }
        )
    return pd.DataFrame(rows)


def _collect_disjoint_dir_splits(
    root: Path | None,
    label: int,
    first_limit: int,
    second_limit: int,
    seed: int,
    first_source: str,
    second_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame(columns=["path", "label", "label_str", "abs_path", "source"])
    if root is None or not root.exists():
        return empty, empty

    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not files:
        return empty, empty

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]

    first_take = files[: max(0, int(first_limit))]
    second_take = files[max(0, int(first_limit)): max(0, int(first_limit) + int(second_limit))]

    def _rows(selected: list[Path], source: str) -> pd.DataFrame:
        if not selected:
            return empty
        rows = []
        for p in selected:
            rows.append(
                {
                    "path": str(p),
                    "label": int(label),
                    "label_str": "real" if int(label) == 1 else "fake",
                    "abs_path": str(p),
                    "source": source,
                }
            )
        return pd.DataFrame(rows)

    return _rows(first_take, first_source), _rows(second_take, second_source)


def _image_quality_ok(path: str) -> bool:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if min(h, w) < 48:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))
    if std < 4.0:
        return False
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap < 1.0:
        return False
    brightness = float(np.mean(gray))
    if brightness < 8.0 or brightness > 247.0:
        return False
    return True


def _apply_quality_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    keep_mask = df["abs_path"].map(_image_quality_ok)
    dropped = int((~keep_mask).sum())
    return df.loc[keep_mask].reset_index(drop=True), dropped


def _append_hard_failure(df: pd.DataFrame, image_path: Path, label: int, copies: int, source: str) -> pd.DataFrame:
    if copies <= 0 or not image_path.exists():
        return df
    rows = []
    for _ in range(copies):
        rows.append(
            {
                "path": str(image_path),
                "label": int(label),
                "label_str": "real" if int(label) == 1 else "fake",
                "abs_path": str(image_path),
                "source": source,
            }
        )
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def _balance_binary(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    real_df = df[df["label"].astype(int) == 1]
    fake_df = df[df["label"].astype(int) == 0]
    if real_df.empty or fake_df.empty:
        return df.reset_index(drop=True)
    n = min(len(real_df), len(fake_df))
    real_bal = real_df.sample(n=n, random_state=seed).reset_index(drop=True)
    fake_bal = fake_df.sample(n=n, random_state=seed + 1).reset_index(drop=True)
    out = pd.concat([real_bal, fake_bal], ignore_index=True).sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return out


def _extract_matrix(df: pd.DataFrame, spec: ClassicalFeatureSpec) -> tuple[np.ndarray, np.ndarray]:
    feats = []
    labels = []
    for _, row in df.iterrows():
        try:
            vec = extract_classical_features_from_path(Path(row["abs_path"]), spec=spec)
            feats.append(vec)
            labels.append(int(row["label"]))
        except Exception:
            continue
    if not feats:
        raise RuntimeError("No features extracted; check dataset paths.")
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _load_cached_feature_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    return np.asarray(payload["X"], dtype=np.float32), np.asarray(payload["y"], dtype=np.int32)


def _feature_cache_paths(cfg, cache_prefix: str) -> tuple[Path, Path, Path]:
    prefix = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(cache_prefix).strip())
    if not prefix:
        return (
            cfg.classical_feature_dir / "train_features.npz",
            cfg.classical_feature_dir / "valid_features.npz",
            cfg.classical_feature_dir / "test_features.npz",
        )
    return (
        cfg.classical_feature_dir / f"{prefix}_train_features.npz",
        cfg.classical_feature_dir / f"{prefix}_valid_features.npz",
        cfg.classical_feature_dir / f"{prefix}_test_features.npz",
    )


def _append_external_hard_examples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hard_dir: Path | None,
    spec: ClassicalFeatureSpec,
    repeat_count: int,
    label: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if hard_dir is None or not hard_dir.exists() or repeat_count <= 0:
        return X_train, y_train, {"source_files": 0, "added_samples": 0, "directory": str(hard_dir) if hard_dir else None}

    feats: list[np.ndarray] = []
    for p in sorted(hard_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            feats.append(extract_classical_features_from_path(p, spec=spec))
        except Exception:
            continue

    if not feats:
        return X_train, y_train, {"source_files": 0, "added_samples": 0, "directory": str(hard_dir)}

    hard_X = np.asarray(feats, dtype=np.float32)
    hard_y = np.full(len(hard_X), label, dtype=np.int32)
    if repeat_count > 1:
        hard_X = np.repeat(hard_X, repeat_count, axis=0)
        hard_y = np.repeat(hard_y, repeat_count, axis=0)

    X_aug = np.concatenate([X_train, hard_X], axis=0)
    y_aug = np.concatenate([y_train, hard_y], axis=0)
    return X_aug, y_aug, {
        "source_files": int(len(feats)),
        "added_samples": int(len(hard_X)),
        "directory": str(hard_dir),
    }


def _fit_with_sample_weight(model: Pipeline, X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None = None) -> None:
    if sample_weight is None:
        model.fit(X_train, y_train)
        return
    try:
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    except TypeError:
        model.fit(X_train, y_train)


def _metrics(y_true: np.ndarray, prob_real: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob_real >= threshold).astype(int)
    real_mask = y_true == 1
    fake_mask = y_true == 0
    fpr_real_as_fake = float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else 0.0
    fnr_fake_as_real = float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else 0.0
    pred_real_ratio = float(np.mean(pred == 1)) if len(pred) else 1.0
    pred_fake_ratio = float(np.mean(pred == 0)) if len(pred) else 1.0
    majority_ratio = max(pred_real_ratio, pred_fake_ratio)
    macro_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "macro_f1": macro_f1,
        "roc_auc": float(roc_auc_score(y_true, prob_real)) if len(np.unique(y_true)) > 1 else 0.5,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "fpr_real_as_fake": fpr_real_as_fake,
        "fnr_fake_as_real": fnr_fake_as_real,
        "real_recall": float(1.0 - fpr_real_as_fake),
        "fake_recall": float(1.0 - fnr_fake_as_real),
        "pred_real_ratio": pred_real_ratio,
        "pred_fake_ratio": pred_fake_ratio,
        "majority_ratio": float(majority_ratio),
        "collapse_rejected": bool(majority_ratio > 0.85),
        "threshold": float(threshold),
    }


def _threshold_objective(m: dict) -> float:
    # Symmetrically prioritize fake_recall and real_recall to prevent real->fake mistakes
    return 0.45 * float(m["fake_recall"]) + 0.40 * float(m["real_recall"]) + 0.15 * float(m["macro_f1"])


def _threshold_sweep(y_true: np.ndarray, prob_real: np.ndarray) -> tuple[float, dict, dict]:
    thresholds = np.linspace(0.30, 0.90, 121)
    best_threshold = 0.5
    best_metrics = _metrics(y_true, prob_real, threshold=0.5)
    best_score = -1e9
    summary = {"tested": []}

    for t in thresholds:
        m = _metrics(y_true, prob_real, threshold=float(t))
        score = _threshold_objective(m)
        row = {
            "threshold": float(t),
            "score": float(score),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "fake_recall": m["fake_recall"],
            "real_recall": m["real_recall"],
            "fnr_fake_as_real": m["fnr_fake_as_real"],
            "majority_ratio": m["majority_ratio"],
            "collapse_rejected": m["collapse_rejected"],
        }
        summary["tested"].append(row)

        if m["collapse_rejected"]:
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_metrics = m
        elif np.isclose(score, best_score):
            # Tie-break toward fewer fake->real mistakes.
            if float(m["fnr_fake_as_real"]) < float(best_metrics["fnr_fake_as_real"]):
                best_threshold = float(t)
                best_metrics = m

    if best_score < -1e8:
        raise RuntimeError("All thresholds failed collapse guard (>85% one-class predictions).")

    summary["selected"] = {
        "threshold": best_threshold,
        "score": float(best_score),
        "objective": "0.45*fake_recall + 0.40*real_recall + 0.15*macro_f1",
    }
    return best_threshold, best_metrics, summary


def _calibration_audit_holdout(y_valid: np.ndarray, prob_real_valid: np.ndarray, seed: int) -> dict:
    # Audit only: if calibration hurts the target objective, keep raw probs.
    if len(np.unique(y_valid)) < 2 or len(y_valid) < 200:
        raw_thr, raw_m, _ = _threshold_sweep(y_valid, prob_real_valid)
        return {
            "mode_scores": {"raw": {"threshold": raw_thr, "metrics": raw_m, "score": _threshold_objective(raw_m)}},
            "best_mode": "raw",
            "enabled": False,
            "reason": "insufficient_validation_for_holdout_calibration",
        }

    idx = np.arange(len(y_valid))
    cal_idx, eval_idx = train_test_split(
        idx,
        test_size=0.40,
        random_state=seed,
        stratify=y_valid,
    )
    p_cal = prob_real_valid[cal_idx]
    y_cal = y_valid[cal_idx]
    p_eval = prob_real_valid[eval_idx]
    y_eval = y_valid[eval_idx]

    # Raw
    raw_thr, raw_m, _ = _threshold_sweep(y_eval, p_eval)
    mode_scores = {"raw": {"threshold": raw_thr, "metrics": raw_m, "score": float(_threshold_objective(raw_m))}}

    # Platt-style 1D logistic map
    try:
        platt = LogisticRegression(max_iter=600, random_state=seed)
        platt.fit(p_cal.reshape(-1, 1), y_cal)
        p_eval_platt = platt.predict_proba(p_eval.reshape(-1, 1))[:, 1]
        platt_thr, platt_m, _ = _threshold_sweep(y_eval, p_eval_platt)
        mode_scores["platt"] = {
            "threshold": platt_thr,
            "metrics": platt_m,
            "score": float(_threshold_objective(platt_m)),
        }
    except Exception:
        pass

    best_mode = max(mode_scores.items(), key=lambda kv: kv[1]["score"])[0]
    enabled = best_mode != "raw"
    reason = "raw_outperformed_calibration" if best_mode == "raw" else "calibration_helped_on_holdout"
    return {
        "mode_scores": mode_scores,
        "best_mode": "raw",
        "enabled": False,  # keep inference simple and deterministic; prefer raw scores for fake sensitivity.
        "reason": reason,
    }


def _mine_hard_examples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    rounds: int,
    repeat_count: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if rounds <= 0 or repeat_count <= 0:
        return X_train, y_train, {"rounds": [], "added_samples": 0}

    idx = np.arange(len(y_train))
    train_idx, mine_idx = train_test_split(
        idx,
        test_size=0.15,
        random_state=seed,
        stratify=y_train,
    )
    core_X = X_train[train_idx]
    core_y = y_train[train_idx]
    mine_X = X_train[mine_idx]
    mine_y = y_train[mine_idx]

    rounds_meta: list[dict] = []
    total_added = 0
    for round_idx in range(rounds):
        miner = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed + round_idx)),
            ]
        )
        miner.fit(core_X, core_y)
        mine_prob = miner.predict_proba(mine_X)[:, 1]
        threshold, metrics, _summary = _threshold_sweep(mine_y, mine_prob)

        hard_mask = ((mine_y == 0) & (mine_prob >= threshold)) | ((mine_y == 1) & (mine_prob < threshold))
        hard_idx = np.where(hard_mask)[0]
        if len(hard_idx) == 0:
            rounds_meta.append(
                {
                    "round": int(round_idx + 1),
                    "threshold": float(threshold),
                    "hard_mistakes": 0,
                    "added_samples": 0,
                }
            )
            break

        dup_X = np.repeat(mine_X[hard_idx], repeat_count, axis=0)
        dup_y = np.repeat(mine_y[hard_idx], repeat_count, axis=0)
        core_X = np.concatenate([core_X, dup_X], axis=0)
        core_y = np.concatenate([core_y, dup_y], axis=0)
        added = int(len(dup_X))
        total_added += added
        rounds_meta.append(
            {
                "round": int(round_idx + 1),
                "threshold": float(threshold),
                "hard_mistakes": int(len(hard_idx)),
                "added_samples": added,
                "mine_fake_recall": float(metrics["fake_recall"]),
                "mine_real_recall": float(metrics["real_recall"]),
            }
        )

    full_X = np.concatenate([core_X, mine_X], axis=0)
    full_y = np.concatenate([core_y, mine_y], axis=0)
    return full_X, full_y, {"rounds": rounds_meta, "added_samples": int(total_added)}


def main() -> None:
    args = parse_args()
    cfg = get_config()
    rng_seed = int(args.seed)
    use_base_csv = bool(int(args.use_base_csv))
    image_root = _resolve_image_root(args.image_root) if use_base_csv else Path(args.image_root)
    spec = ClassicalFeatureSpec()

    quality_filter_enabled = bool(int(args.quality_filter))
    hard_failure_path = Path(args.hard_failure_image).resolve()

    if use_base_csv:
        # Base Kaggle splits
        train_df = _load_split(cfg.train_csv, image_root=image_root, limit=args.max_train, seed=rng_seed)
        valid_df = _load_split(cfg.valid_csv, image_root=image_root, limit=args.max_valid, seed=rng_seed + 1)
        test_df = _load_split(cfg.test_csv, image_root=image_root, limit=args.max_test, seed=rng_seed + 2)

        # Extra synthetic-style data (rvf10k)
        extra_train_real_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_train_real_root),
            label=1,
            limit=int(args.extra_train_per_class),
            seed=rng_seed + 10,
            source="rvf10k_train_real",
        )
        extra_train_fake_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_train_fake_root),
            label=0,
            limit=int(args.extra_train_per_class),
            seed=rng_seed + 11,
            source="rvf10k_train_fake",
        )
        extra_valid_real_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_valid_real_root),
            label=1,
            limit=int(args.extra_valid_per_class),
            seed=rng_seed + 12,
            source="rvf10k_valid_real",
        )
        extra_valid_fake_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_valid_fake_root),
            label=0,
            limit=int(args.extra_valid_per_class),
            seed=rng_seed + 13,
            source="rvf10k_valid_fake",
        )

        train_df = pd.concat([train_df, extra_train_real_df, extra_train_fake_df], ignore_index=True)
        valid_df = pd.concat([valid_df, extra_valid_real_df, extra_valid_fake_df], ignore_index=True)
    else:
        local_train_real_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_train_real_root),
            label=1,
            limit=int(args.local_train_per_class),
            seed=rng_seed + 10,
            source="local_train_real",
        )
        local_train_fake_df = _collect_dir_split(
            _resolve_optional_dir(args.extra_train_fake_root),
            label=0,
            limit=int(args.local_train_per_class),
            seed=rng_seed + 11,
            source="local_train_fake",
        )
        local_valid_real_df, local_test_real_df = _collect_disjoint_dir_splits(
            _resolve_optional_dir(args.extra_valid_real_root),
            label=1,
            first_limit=int(args.local_valid_per_class),
            second_limit=int(args.local_test_per_class),
            seed=rng_seed + 12,
            first_source="local_valid_real",
            second_source="local_test_real",
        )
        local_valid_fake_df, local_test_fake_df = _collect_disjoint_dir_splits(
            _resolve_optional_dir(args.extra_valid_fake_root),
            label=0,
            first_limit=int(args.local_valid_per_class),
            second_limit=int(args.local_test_per_class),
            seed=rng_seed + 13,
            first_source="local_valid_fake",
            second_source="local_test_fake",
        )
        train_df = pd.concat([local_train_real_df, local_train_fake_df], ignore_index=True)
        valid_df = pd.concat([local_valid_real_df, local_valid_fake_df], ignore_index=True)
        test_df = pd.concat([local_test_real_df, local_test_fake_df], ignore_index=True)

    train_df = _append_hard_failure(
        train_df,
        image_path=hard_failure_path,
        label=0,
        copies=int(args.hard_failure_train_copies),
        source="hard_failure_f3_train",
    )
    valid_df = _append_hard_failure(
        valid_df,
        image_path=hard_failure_path,
        label=0,
        copies=int(args.hard_failure_valid_copies),
        source="hard_failure_f3_valid",
    )

    dropped_quality = {"train": 0, "valid": 0, "test": 0}
    if quality_filter_enabled:
        if use_base_csv:
            # Keep Kaggle core split untouched; clean supplemental synthetic sources.
            train_base = train_df[train_df["source"] == "kaggle_140k"]
            train_extra = train_df[train_df["source"] != "kaggle_140k"]
            train_extra, dropped_quality["train"] = _apply_quality_filter(train_extra)
            train_df = pd.concat([train_base, train_extra], ignore_index=True)

            valid_base = valid_df[valid_df["source"] == "kaggle_140k"]
            valid_extra = valid_df[valid_df["source"] != "kaggle_140k"]
            valid_extra, dropped_quality["valid"] = _apply_quality_filter(valid_extra)
            valid_df = pd.concat([valid_base, valid_extra], ignore_index=True)
        else:
            train_df, dropped_quality["train"] = _apply_quality_filter(train_df)
            valid_df, dropped_quality["valid"] = _apply_quality_filter(valid_df)
            test_df, dropped_quality["test"] = _apply_quality_filter(test_df)

    # Strict class balance (clean + balanced objective)
    train_df = _balance_binary(train_df, seed=rng_seed + 30)
    valid_df = _balance_binary(valid_df, seed=rng_seed + 31)
    test_df = _balance_binary(test_df, seed=rng_seed + 32)

    cfg.classical_feature_dir.mkdir(parents=True, exist_ok=True)
    train_features_path, valid_features_path, test_features_path = _feature_cache_paths(cfg, args.cache_prefix)
    reuse_features = bool(int(args.reuse_features))
    can_reuse = reuse_features and train_features_path.exists() and valid_features_path.exists() and test_features_path.exists()

    if can_reuse:
        X_train, y_train = _load_cached_feature_matrix(train_features_path)
        X_valid, y_valid = _load_cached_feature_matrix(valid_features_path)
        X_test, y_test = _load_cached_feature_matrix(test_features_path)
    else:
        X_train, y_train = _extract_matrix(train_df, spec)
        X_valid, y_valid = _extract_matrix(valid_df, spec)
        X_test, y_test = _extract_matrix(test_df, spec)
        np.savez_compressed(train_features_path, X=X_train, y=y_train)
        np.savez_compressed(valid_features_path, X=X_valid, y=y_valid)
        np.savez_compressed(test_features_path, X=X_test, y=y_test)

    X_train_aug, y_train_aug, hard_mine_meta = _mine_hard_examples(
        X_train,
        y_train,
        seed=rng_seed + 200,
        rounds=int(args.hard_mine_rounds),
        repeat_count=int(args.hard_mine_repeat),
    )
    external_hard_fake_dir = _resolve_optional_dir(args.external_hard_fake_dir)
    X_train_aug, y_train_aug, external_hard_fake_meta = _append_external_hard_examples(
        X_train_aug,
        y_train_aug,
        hard_dir=external_hard_fake_dir,
        spec=spec,
        repeat_count=int(args.external_hard_fake_repeat),
        label=0,
    )
    external_hard_real_dir = _resolve_optional_dir(args.external_hard_real_dir)
    X_train_aug, y_train_aug, external_hard_real_meta = _append_external_hard_examples(
        X_train_aug,
        y_train_aug,
        hard_dir=external_hard_real_dir,
        spec=spec,
        repeat_count=int(args.external_hard_real_repeat),
        label=1,
    )
    train_sample_weight = np.ones(len(y_train_aug), dtype=np.float32)
    train_sample_weight[y_train_aug == 0] = float(args.fake_sample_weight)

    models: dict[str, Pipeline] = {
        "xgboost": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=260,
                        max_depth=6,
                        learning_rate=0.06,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=rng_seed,
                        tree_method="hist",
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=rng_seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("clf", RandomForestClassifier(n_estimators=380, max_depth=26, random_state=rng_seed, n_jobs=-1, class_weight="balanced_subsample")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=700, class_weight="balanced", random_state=rng_seed)),
            ]
        ),
    }

    valid_metrics: dict[str, dict] = {}
    test_metrics: dict[str, dict] = {}
    threshold_meta: dict[str, dict] = {}
    calibration_audit: dict[str, dict] = {}
    model_valid_prob: dict[str, np.ndarray] = {}
    model_test_prob: dict[str, np.ndarray] = {}

    for name, model in models.items():
        _fit_with_sample_weight(model, X_train_aug, y_train_aug, sample_weight=train_sample_weight)
        p_valid = model.predict_proba(X_valid)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]
        model_valid_prob[name] = p_valid
        model_test_prob[name] = p_test

        selected_threshold, selected_valid_metrics, sweep_summary = _threshold_sweep(y_valid, p_valid)
        valid_metrics[name] = selected_valid_metrics
        test_metrics[name] = _metrics(y_test, p_test, threshold=selected_threshold)
        threshold_meta[name] = sweep_summary
        calibration_audit[name] = _calibration_audit_holdout(y_valid, p_valid, seed=rng_seed + 100)

    ranked = sorted(
        valid_metrics.items(),
        key=lambda kv: (
            kv[1]["fake_recall"],
            _threshold_objective(kv[1]),
            kv[1]["balanced_accuracy"],
            kv[1]["macro_f1"],
            -kv[1]["majority_ratio"],
        ),
        reverse=True,
    )
    best_name = ranked[0][0]
    best_model = models[best_name]
    joblib.dump(best_model, cfg.classical_model_path)

    metadata = {
        "backend": "classical_fallback",
        "best_model": best_name,
        "decision_threshold": float(valid_metrics[best_name]["threshold"]),
        "image_root": str(image_root),
        "feature_spec": {
            "color_hist_bins": spec.color_hist_bins,
            "gray_hist_bins": spec.gray_hist_bins,
            "lbp_bins": spec.lbp_bins,
            "resize_side": spec.resize_side,
        },
        "threshold_objective": "0.45*fake_recall + 0.40*real_recall + 0.15*macro_f1",
        "quality_filter_enabled": quality_filter_enabled,
        "use_base_csv": use_base_csv,
        "cache_prefix": str(args.cache_prefix),
        "resumed_from_feature_cache": bool(can_reuse),
        "quality_filter_dropped": dropped_quality,
        "split_counts": {"train": int(len(y_train_aug)), "valid": int(len(y_valid)), "test": int(len(y_test))},
        "dataset_counts": {
            "train_real": int(np.sum(y_train_aug == 1)),
            "train_fake": int(np.sum(y_train_aug == 0)),
            "valid_real": int(np.sum(y_valid == 1)),
            "valid_fake": int(np.sum(y_valid == 0)),
            "test_real": int(np.sum(y_test == 1)),
            "test_fake": int(np.sum(y_test == 0)),
        },
        "training_bias": {
            "fake_sample_weight": float(args.fake_sample_weight),
            "hard_failure_loop": hard_mine_meta,
            "external_hard_fake": external_hard_fake_meta,
            "external_hard_real": external_hard_real_meta,
        },
        "sources": {
            "train": train_df["source"].value_counts().to_dict(),
            "valid": valid_df["source"].value_counts().to_dict(),
            "test": test_df["source"].value_counts().to_dict(),
            "hard_failure_image": str(hard_failure_path) if hard_failure_path.exists() else None,
        },
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "threshold_sweep": threshold_meta,
        "calibration_audit": calibration_audit,
        "classical_calibrator": {
            "enabled": False,
            "selected_mode": calibration_audit.get(best_name, {}).get("best_mode", "raw"),
            "reason": calibration_audit.get(best_name, {}).get("reason", "raw_outperformed_calibration"),
        },
        "collapse_guard": {
            "max_one_class_ratio": 0.85,
            "best_model_validation_majority_ratio": float(valid_metrics[best_name]["majority_ratio"]),
            "passed": bool(valid_metrics[best_name]["majority_ratio"] <= 0.85),
        },
    }
    cfg.classical_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (cfg.results_dir / "classical_metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    prod = {"backend": "classical_fallback", "model": best_name}
    cfg.production_inference_path.write_text(json.dumps(prod, indent=2), encoding="utf-8")

    # Save 10 real + 10 fake sample predictions from test split.
    test_probs = model_test_prob[best_name]
    selected_threshold = float(valid_metrics[best_name]["threshold"])
    test_pred = (test_probs >= selected_threshold).astype(int)
    out_rows = []
    for label_target in [1, 0]:
        idx = np.where(y_test == label_target)[0][:10]
        for i in idx:
            out_rows.append(
                {
                    "abs_path": test_df.iloc[i]["abs_path"],
                    "true_label": "real" if int(y_test[i]) == 1 else "fake",
                    "predicted_label": "real" if int(test_pred[i]) == 1 else "fake",
                    "prob_real": float(test_probs[i]),
                    "prob_fake": float(1.0 - test_probs[i]),
                }
            )
    pd.DataFrame(out_rows).to_csv(cfg.results_dir / "classical_sample_predictions.csv", index=False)

    print(
        json.dumps(
            {
                "best_model": best_name,
                "decision_threshold": selected_threshold,
                "model_path": str(cfg.classical_model_path),
                "metadata_path": str(cfg.classical_metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
