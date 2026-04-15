from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)


@dataclass
class ProbabilityCalibrator:
    model: LogisticRegression

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        try:
            calibrated = self.model.predict_proba(probs.reshape(-1, 1))[:, 1]
            return np.clip(calibrated, 1e-6, 1.0 - 1e-6)
        except Exception as exc:  # pylint: disable=broad-except
            # Pickle from a different sklearn major often breaks internal LR state (e.g. missing multi_class).
            LOGGER.warning(
                'Calibrator predict_proba failed (%s); returning clipped input probabilities.',
                exc,
            )
            return probs


@dataclass
class IsotonicCalibrator:
    model: IsotonicRegression

    def transform(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        try:
            calibrated = self.model.predict(probs.reshape(-1))
            return np.clip(calibrated.astype(np.float64), 1e-6, 1.0 - 1e-6)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('Isotonic calibrator failed (%s); returning clipped input probabilities.', exc)
            return probs


def fit_platt_calibrator(y_true: np.ndarray, probs: np.ndarray) -> ProbabilityCalibrator:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(probs.reshape(-1, 1), y_true)
    return ProbabilityCalibrator(model=model)


def fit_isotonic_calibrator(y_true: np.ndarray, probs: np.ndarray) -> IsotonicCalibrator:
    y_true = np.asarray(y_true).astype(np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    model.fit(probs, y_true)
    return IsotonicCalibrator(model=model)


def fit_hybrid_calibrator_isotonic(
    y_true: np.ndarray,
    probs: np.ndarray,
    min_samples: int = 200,
) -> IsotonicCalibrator | ProbabilityCalibrator:
    """Prefer isotonic mapping for hybrid XGBoost scores; fall back to Platt if too few points."""
    y_true = np.asarray(y_true)
    if len(y_true) >= min_samples:
        return fit_isotonic_calibrator(y_true=y_true, probs=probs)
    LOGGER.warning(
        'Validation size %d < %d; using Platt (sigmoid) calibrator instead of isotonic.',
        len(y_true),
        min_samples,
    )
    return fit_platt_calibrator(y_true=y_true, probs=probs)


def apply_calibration(
    probs: np.ndarray,
    calibrator: ProbabilityCalibrator | IsotonicCalibrator | None,
) -> tuple[np.ndarray, bool]:
    """Return calibrated probabilities and whether a calibrator was applied successfully."""
    probs = np.asarray(probs, dtype=np.float64)
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    if calibrator is None:
        return clipped, False
    try:
        return calibrator.transform(clipped), True
    except Exception as exc:  # pylint: disable=broad-except
        # Joblib may load on a different sklearn version than training; fall back safely.
        LOGGER.warning(
            'Probability calibrator failed (%s). Using uncalibrated scores. '
            'Align scikit-learn with the training environment (see requirements.txt).',
            exc,
        )
        return clipped, False


def save_calibrator(calibrator: ProbabilityCalibrator | IsotonicCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, path)


def load_calibrator(path: Path) -> ProbabilityCalibrator | IsotonicCalibrator:
    if not path.exists():
        raise FileNotFoundError(f'Calibrator not found: {path}')
    obj = joblib.load(path)
    if not isinstance(obj, (ProbabilityCalibrator, IsotonicCalibrator)):
        raise TypeError(f'Invalid calibrator object in {path}')
    return obj


def load_optional_calibrator(path: Path) -> ProbabilityCalibrator | IsotonicCalibrator | None:
    if not path.exists():
        return None
    try:
        return load_calibrator(path)
    except Exception:
        return None
