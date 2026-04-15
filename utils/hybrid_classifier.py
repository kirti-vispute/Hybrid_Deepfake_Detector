from __future__ import annotations

import numpy as np


class SoftVotingHybridEnsemble:
    """Simple calibrated soft-voting ensemble for binary classifiers."""

    def __init__(self, models: list, weights: list[float] | None = None) -> None:
        if not models:
            raise ValueError('At least one model is required for ensemble.')
        self.models = models
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError('weights length must match models length.')
        arr = np.asarray(weights, dtype=np.float64)
        if np.any(arr < 0):
            raise ValueError('weights must be non-negative.')
        if np.sum(arr) <= 0:
            raise ValueError('sum(weights) must be > 0.')
        self.weights = (arr / np.sum(arr)).tolist()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for model in self.models:
            proba = model.predict_proba(x)
            if proba.ndim != 2 or proba.shape[1] != 2:
                raise ValueError('Expected predict_proba output shape (N, 2).')
            probs.append(proba)
        stacked = np.stack(probs, axis=0)
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1, 1, 1)
        blended = np.sum(stacked * weights, axis=0)
        blended = np.clip(blended, 1e-6, 1.0 - 1e-6)
        blended /= blended.sum(axis=1, keepdims=True)
        return blended

