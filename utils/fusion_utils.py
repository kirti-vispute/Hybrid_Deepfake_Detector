from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def macro_f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    f1_real = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    return float((f1_real + f1_fake) / 2.0)


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> tuple[float, dict]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if thresholds is None:
        # Prefer 0.55–0.70 to reduce "always fake" bias from poorly calibrated scores.
        thresholds = np.linspace(0.55, 0.70, 61)

    best_threshold = 0.5
    best_score = -1.0
    best_accuracy = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = macro_f1_binary(y_true, y_pred)
        acc = float(accuracy_score(y_true, y_pred))
        if score > best_score or (np.isclose(score, best_score) and acc > best_accuracy):
            best_score = score
            best_accuracy = acc
            best_threshold = float(threshold)

    if best_score <= 0.0 and len(thresholds) < 100:
        return optimize_threshold(y_true, y_prob, thresholds=np.linspace(0.2, 0.85, 131))

    return best_threshold, {
        'macro_f1': float(best_score),
        'accuracy': float(best_accuracy),
        'threshold': float(best_threshold),
    }


def apply_weighted_blend(cnn_prob: np.ndarray, hybrid_prob: np.ndarray, cnn_weight: float) -> np.ndarray:
    cnn_prob = np.asarray(cnn_prob, dtype=np.float64)
    hybrid_prob = np.asarray(hybrid_prob, dtype=np.float64)
    w = float(np.clip(cnn_weight, 0.0, 1.0))
    return np.clip(w * cnn_prob + (1.0 - w) * hybrid_prob, 1e-6, 1.0 - 1e-6)


@dataclass
class ClassAwareFusionParams:
    cnn_weight: float
    real_gate: float
    fake_gate: float
    decision_threshold: float


def apply_class_aware_fusion(cnn_prob: np.ndarray, hybrid_prob: np.ndarray, params: ClassAwareFusionParams) -> np.ndarray:
    cnn_prob = np.asarray(cnn_prob, dtype=np.float64)
    hybrid_prob = np.asarray(hybrid_prob, dtype=np.float64)

    blended = apply_weighted_blend(cnn_prob, hybrid_prob, params.cnn_weight)

    cnn_real_mask = cnn_prob >= params.real_gate
    hybrid_fake_mask = hybrid_prob <= params.fake_gate

    both_mask = cnn_real_mask & hybrid_fake_mask
    only_real_mask = cnn_real_mask & ~hybrid_fake_mask
    only_fake_mask = hybrid_fake_mask & ~cnn_real_mask

    out = blended.copy()
    out[only_real_mask] = cnn_prob[only_real_mask]
    out[only_fake_mask] = hybrid_prob[only_fake_mask]

    if np.any(both_mask):
        cnn_margin = np.abs(cnn_prob[both_mask] - 0.5)
        hybrid_margin = np.abs(hybrid_prob[both_mask] - 0.5)
        choose_cnn = cnn_margin >= hybrid_margin
        choose_hybrid = ~choose_cnn
        both_idx = np.where(both_mask)[0]
        out[both_idx[choose_cnn]] = cnn_prob[both_idx[choose_cnn]]
        out[both_idx[choose_hybrid]] = hybrid_prob[both_idx[choose_hybrid]]

    return np.clip(out, 1e-6, 1.0 - 1e-6)


def fit_class_aware_fusion(
    y_true: np.ndarray,
    cnn_prob: np.ndarray,
    hybrid_prob: np.ndarray,
    cnn_weights: list[float] | None = None,
    real_gates: list[float] | None = None,
    fake_gates: list[float] | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    cnn_prob = np.asarray(cnn_prob, dtype=np.float64)
    hybrid_prob = np.asarray(hybrid_prob, dtype=np.float64)

    if cnn_weights is None:
        cnn_weights = [0.25, 0.35, 0.5, 0.65, 0.75]
    if real_gates is None:
        real_gates = [0.55, 0.6, 0.65, 0.7, 0.75]
    if fake_gates is None:
        fake_gates = [0.45, 0.4, 0.35, 0.3, 0.25]
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 121)

    best = {
        'macro_f1': -1.0,
        'accuracy': -1.0,
        'params': None,
    }

    for cnn_weight, real_gate, fake_gate in itertools.product(cnn_weights, real_gates, fake_gates):
        params = ClassAwareFusionParams(
            cnn_weight=float(cnn_weight),
            real_gate=float(real_gate),
            fake_gate=float(fake_gate),
            decision_threshold=0.5,
        )
        fused_prob = apply_class_aware_fusion(cnn_prob=cnn_prob, hybrid_prob=hybrid_prob, params=params)
        threshold, threshold_stats = optimize_threshold(y_true=y_true, y_prob=fused_prob, thresholds=thresholds)

        macro_f1 = threshold_stats['macro_f1']
        acc = threshold_stats['accuracy']

        if macro_f1 > best['macro_f1'] or (np.isclose(macro_f1, best['macro_f1']) and acc > best['accuracy']):
            params.decision_threshold = float(threshold)
            best = {
                'macro_f1': float(macro_f1),
                'accuracy': float(acc),
                'params': {
                    'cnn_weight': params.cnn_weight,
                    'real_gate': params.real_gate,
                    'fake_gate': params.fake_gate,
                    'decision_threshold': params.decision_threshold,
                },
            }

    return best
