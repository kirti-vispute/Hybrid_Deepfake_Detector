"""Threshold selection: balance TPR and TNR (approximate equality)."""
from __future__ import annotations

import numpy as np


def balanced_tpr_tnr_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    """
    Search thresholds where TPR ≈ TNR to reduce one-sided false real/fake rates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.linspace(0.01, 0.99, 500)
    best_t = 0.5
    best_diff = float('inf')
    best_tpr = 0.0
    best_tnr = 0.0

    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5, {'method': 'balanced_tpr_tnr', 'note': 'degenerate_class', 'threshold': 0.5}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        diff = abs(tpr - tnr)
        if diff < best_diff:
            best_diff = diff
            best_t = float(t)
            best_tpr = float(tpr)
            best_tnr = float(tnr)

    return best_t, {
        'method': 'balanced_tpr_tnr',
        'tpr': best_tpr,
        'tnr': best_tnr,
        'threshold': best_t,
        'abs_tpr_tnr_gap': float(best_diff),
    }
