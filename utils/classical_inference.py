from __future__ import annotations

import json
from io import BytesIO

import cv2
import joblib
import numpy as np
from PIL import Image

from utils.classical_features import ClassicalFeatureSpec, extract_classical_features_from_bgr
from utils.config import AppConfig


def load_classical_bundle(config: AppConfig) -> dict:
    if not config.classical_model_path.exists():
        raise FileNotFoundError(f"Missing classical model artifact: {config.classical_model_path}")
    if not config.classical_metadata_path.exists():
        raise FileNotFoundError(f"Missing classical metadata artifact: {config.classical_metadata_path}")
    model = None
    load_error = None
    try:
        model = joblib.load(config.classical_model_path)
        # Keep inference deterministic and avoid thread oversubscription stalls.
        if hasattr(model, "set_params"):
            try:
                model.set_params(clf__n_jobs=1)
            except Exception:
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass
    except Exception as exc:  # pylint: disable=broad-except
        load_error = str(exc)

    metadata = json.loads(config.classical_metadata_path.read_text(encoding="utf-8-sig"))

    # Parse ClassicalFeatureSpec once at load time — avoids per-request dict parsing.
    fspec_raw = metadata.get("feature_spec", {})
    feat_spec = ClassicalFeatureSpec(
        color_hist_bins=int(fspec_raw.get("color_hist_bins", 16)),
        gray_hist_bins=int(fspec_raw.get("gray_hist_bins", 32)),
        lbp_bins=int(fspec_raw.get("lbp_bins", 32)),
        resize_side=int(fspec_raw.get("resize_side", 128)),
    )

    return {
        "model": model,
        "metadata": metadata,
        "load_error": load_error,
        "feat_spec": feat_spec,  # cached — avoids repeated dict lookups per request
    }


def _predict_heuristic_real_probability(bgr: np.ndarray) -> float:
    """Lightweight, dependency-safe fallback when pickled sklearn artifacts cannot load."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_sigma = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))
    sat_mean = float(np.mean(hsv[:, :, 1]))

    sharpness_norm = float(np.clip((lap_var - 40.0) / 260.0, 0.0, 1.0))
    noise_norm = float(np.clip((noise_sigma - 2.0) / 18.0, 0.0, 1.0))
    saturation_norm = float(np.clip(sat_mean / 255.0, 0.0, 1.0))

    prob_real = 0.45 * sharpness_norm + 0.35 * noise_norm + 0.20 * saturation_norm
    return float(np.clip(prob_real, 0.05, 0.95))


def _build_result(
    prob_real: float,
    threshold: float,
    model_used: str,
    metadata: dict,
    load_error: str | None = None,
) -> dict:
    """Assemble the standard prediction result dict from a computed probability."""
    pred_label = 1 if prob_real >= threshold else 0
    output = {
        "predicted_class": "Real" if pred_label == 1 else "Fake",
        "confidence": prob_real if pred_label == 1 else 1.0 - prob_real,
        "probabilities": {"real": prob_real, "fake": 1.0 - prob_real},
        "model_used": model_used,
        "inference_backend": model_used,
        "decision_threshold": threshold,
        "calibrated": bool(metadata.get("classical_calibrator", {}).get("enabled", False)),
        "calibrator_mode": metadata.get("classical_calibrator", {}).get("selected_mode", "raw"),
    }
    if load_error:
        output["fallback_warning"] = f"classical_model_unpickle_failed: {load_error}"
    return output


def predict_classical_from_pil(
    pil_image: Image.Image,
    config: AppConfig,
    bundle: dict | None = None,
) -> dict:
    """Predict from an already-decoded PIL image.

    This is the preferred entry point when a PIL image is already available
    (e.g. from the request decode stage) because it avoids opening the raw
    bytes a second time.  ``predict_classical_from_bytes`` delegates here.
    """
    if bundle is None:
        bundle = load_classical_bundle(config)

    model = bundle["model"]
    metadata = bundle["metadata"]
    threshold = float(metadata.get("decision_threshold", 0.5))
    # Use the spec cached at bundle-load time instead of re-parsing on every call.
    spec: ClassicalFeatureSpec = bundle.get("feat_spec") or ClassicalFeatureSpec()

    # Single PIL→BGR conversion (no second Image.open from bytes).
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    if model is None:
        prob_real = _predict_heuristic_real_probability(bgr)
        model_used = "classical_heuristic_fallback"
        load_error = bundle.get("load_error")
    else:
        features = extract_classical_features_from_bgr(bgr, spec=spec).reshape(1, -1)
        prob_real = float(model.predict_proba(features)[0, 1])
        model_used = "classical_fallback"
        load_error = None

    return _build_result(prob_real, threshold, model_used, metadata, load_error)


def predict_classical_from_bytes(
    image_bytes: bytes,
    config: AppConfig,
    bundle: dict | None = None,
) -> dict:
    """Predict from raw image bytes.

    Prefer ``predict_classical_from_pil`` when a PIL image is already
    available upstream — this function exists for backward compatibility
    and standalone CLI use.
    """
    if bundle is None:
        bundle = load_classical_bundle(config)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return predict_classical_from_pil(pil_image, config, bundle)

