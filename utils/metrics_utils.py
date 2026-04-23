from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for idx in range(bins):
        low, high = bin_edges[idx], bin_edges[idx + 1]
        mask = (y_prob >= low) & (y_prob < high if idx < bins - 1 else y_prob <= high)
        if not np.any(mask):
            continue

        bucket_true = y_true[mask]
        bucket_prob = y_prob[mask]
        bucket_acc = np.mean(bucket_true == (bucket_prob >= 0.5).astype(int))
        bucket_conf = np.mean(np.maximum(bucket_prob, 1.0 - bucket_prob))
        ece += (bucket_true.size / y_true.size) * abs(bucket_acc - bucket_conf)

    return float(ece)


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def confidence_statistics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    confidence = np.where(y_pred == 1, y_prob, 1.0 - y_prob)

    correct_mask = y_pred == y_true
    wrong_mask = ~correct_mask

    return {
        'mean_confidence': float(np.mean(confidence)),
        'std_confidence': float(np.std(confidence)),
        'mean_confidence_correct': _safe_mean(confidence[correct_mask]),
        'mean_confidence_incorrect': _safe_mean(confidence[wrong_mask]),
    }


def per_class_confidence_statistics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    confidence = np.where(y_pred == 1, y_prob, 1.0 - y_prob)

    correct_real_mask = (y_true == 1) & (y_pred == 1)
    correct_fake_mask = (y_true == 0) & (y_pred == 0)
    wrong_mask = y_true != y_pred

    return {
        'mean_confidence_correct_real': _safe_mean(confidence[correct_real_mask]),
        'mean_confidence_correct_fake': _safe_mean(confidence[correct_fake_mask]),
        'mean_confidence_wrong': _safe_mean(confidence[wrong_mask]),
        'correct_real_count': int(np.sum(correct_real_mask)),
        'correct_fake_count': int(np.sum(correct_fake_mask)),
        'wrong_count': int(np.sum(wrong_mask)),
    }


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    def _metrics_for_label(label: int) -> dict:
        return {
            'precision': float(precision_score(y_true, y_pred, pos_label=label, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, pos_label=label, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, pos_label=label, zero_division=0)),
            'support': int(np.sum(y_true == label)),
        }

    return {
        'real': _metrics_for_label(1),
        'fake': _metrics_for_label(0),
    }


def false_positive_rate_real(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Fraction of true REAL images (label 1) predicted as FAKE (0). Primary driver for real→fake bias."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    real_mask = y_true == 1
    denom = int(np.sum(real_mask))
    if denom == 0:
        return None
    fp = int(np.sum(real_mask & (y_pred == 0)))
    return float(fp / denom)


def false_negative_rate_fake(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """Fraction of true FAKE images (label 0) predicted as REAL (1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fake_mask = y_true == 0
    denom = int(np.sum(fake_mask))
    if denom == 0:
        return None
    fn = int(np.sum(fake_mask & (y_pred == 1)))
    return float(fn / denom)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        'brier_score': float(brier_score_loss(y_true, y_prob)),
        'ece': expected_calibration_error(y_true=y_true, y_prob=y_prob, bins=10),
        'false_positive_rate_real': false_positive_rate_real(y_true, y_pred),
        'false_negative_rate_fake': false_negative_rate_fake(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'threshold': float(threshold),
        'confidence_stats': confidence_statistics(y_true=y_true, y_prob=y_prob, threshold=threshold),
        'per_class': per_class_prf(y_true=y_true, y_pred=y_pred),
        'confidence_breakdown': per_class_confidence_statistics(y_true=y_true, y_prob=y_prob, threshold=threshold),
    }
    return metrics


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label',
    )

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f'{cm[i, j]}',
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
