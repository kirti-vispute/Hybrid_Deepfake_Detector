from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.calibration_utils import ProbabilityCalibrator, apply_calibration, load_optional_calibrator
from utils.config import AppConfig
from utils.model_utils import (
    get_feature_extractor,
    get_model_input_size,
    load_cnn_model,
    load_optional_joblib,
    load_optional_json,
    load_xgb_model,
    normalize_model_type,
    predict_cnn_real_probs,
    preprocess_single_image_with_meta,
    probability_to_prediction,
)


def _sanitize_preproc(meta: dict) -> dict:
    out: dict = {}
    for key, val in meta.items():
        if key in {'box', 'crop'} and val is not None and isinstance(val, (list, tuple)):
            out[key] = [int(x) for x in val]
        elif isinstance(val, (np.floating, float)):
            out[key] = float(val)
        elif isinstance(val, (np.integer, int)) and key not in {'face_detected'}:
            out[key] = int(val)
        else:
            out[key] = val
    return out


def _predict_prob_cnn(
    batch: np.ndarray,
    cnn_model,
) -> float:
    pred = predict_cnn_real_probs(cnn_model, batch, verbose=0)
    return float(pred.ravel()[0])


def _predict_prob_hybrid(
    batch: np.ndarray,
    cnn_model,
    xgb_model,
    feature_scaler=None,
    pca=None,
) -> float:
    extractor = get_feature_extractor(cnn_model)
    features = extractor.predict(batch, verbose=0)
    if feature_scaler is not None:
        features = feature_scaler.transform(features)
    if pca is not None:
        features = pca.transform(features)
    probabilities = xgb_model.predict_proba(features)
    return float(probabilities[:, 1][0])


def _resolve_model_runtime_settings(config: AppConfig, cnn_model) -> tuple[str, int]:
    metadata = load_optional_json(config.cnn_metadata_path) or {}
    backbone = metadata.get('backbone_name', metadata.get('model_type', config.backbone_name or config.model_type))
    model_type = normalize_model_type(str(backbone).lower())
    image_size = int(metadata.get('image_size', get_model_input_size(cnn_model)))
    return model_type, image_size


def _load_threshold_from_metadata(path: Path, default: float = 0.5) -> float:
    metadata = load_optional_json(path) or {}
    try:
        return float(metadata.get('decision_threshold', default))
    except (TypeError, ValueError):
        return float(default)


def _production_inference_backend(config: AppConfig) -> str:
    payload = load_optional_json(config.production_inference_path) or {}
    backend = str(payload.get('backend', 'hybrid')).lower().strip()
    if backend in {'cnn', 'cnn_direct'}:
        return 'cnn_direct'
    return 'hybrid'


def predict_pil_image(
    image: Image.Image,
    model_choice: str,
    config: AppConfig,
    cnn_model=None,
    xgb_model=None,
    feature_scaler=None,
    cnn_calibrator: ProbabilityCalibrator | None = None,
    hybrid_calibrator: ProbabilityCalibrator | None = None,
) -> dict:
    model_choice = model_choice.lower().strip()
    if model_choice in {'cnn', 'smart'}:
        model_choice = 'hybrid'
    if model_choice != 'hybrid':
        raise ValueError("Invalid model type. Use 'hybrid'.")

    cnn_model = cnn_model or load_cnn_model(config.cnn_model_path)
    model_type, image_size = _resolve_model_runtime_settings(config=config, cnn_model=cnn_model)

    batch, pre_meta = preprocess_single_image_with_meta(image, image_size, model_type)
    pre_meta = _sanitize_preproc(pre_meta)

    hybrid_meta = load_optional_json(config.hybrid_metadata_path) or {}
    try:
        hybrid_threshold = float(hybrid_meta.get('decision_threshold', 0.5))
    except (TypeError, ValueError):
        hybrid_threshold = 0.5
    use_platt_hybrid = bool(hybrid_meta.get('use_platt_overlay', True))
    cnn_threshold = _load_threshold_from_metadata(config.cnn_metadata_path, default=0.5)

    raw_prob_real_cnn = _predict_prob_cnn(batch, cnn_model)
    cnn_cal = cnn_calibrator if cnn_calibrator is not None else load_optional_calibrator(config.cnn_calibrator_path)
    cnn_probs, cnn_cal_applied = apply_calibration(np.array([raw_prob_real_cnn]), cnn_cal)
    prob_real_cnn = float(cnn_probs[0])

    backend = _production_inference_backend(config)
    if backend == 'cnn_direct':
        payload = probability_to_prediction(prob_real_cnn, threshold=cnn_threshold)
        payload['model_used'] = 'cnn'
        payload['calibrated'] = cnn_cal_applied
        payload['inference_backend'] = 'cnn_direct'
        payload['raw_probabilities'] = {
            'real': float(raw_prob_real_cnn),
            'fake': float(1.0 - raw_prob_real_cnn),
        }
        payload['component_probabilities'] = {'cnn_real': prob_real_cnn}
        payload['preprocessing'] = pre_meta
        return payload

    xgb_model = xgb_model or load_xgb_model(config.xgb_model_path)
    scaler = feature_scaler if feature_scaler is not None else load_optional_joblib(config.xgb_scaler_path)
    pca_path = hybrid_meta.get('pca_path')
    pca = load_optional_joblib(Path(pca_path)) if pca_path else load_optional_joblib(config.hybrid_pca_path)

    raw_prob_real_hybrid = _predict_prob_hybrid(batch, cnn_model, xgb_model, feature_scaler=scaler, pca=pca)
    if use_platt_hybrid:
        hybrid_cal = hybrid_calibrator if hybrid_calibrator is not None else load_optional_calibrator(config.hybrid_calibrator_path)
        hybrid_probs, hybrid_cal_applied = apply_calibration(np.array([raw_prob_real_hybrid]), hybrid_cal)
        prob_real_hybrid = float(hybrid_probs[0])
    else:
        prob_real_hybrid = float(raw_prob_real_hybrid)
        hybrid_cal_applied = False

    payload = probability_to_prediction(
        prob_real_hybrid,
        threshold=hybrid_threshold,
        uncertain_low=config.uncertain_prob_low,
        uncertain_high=config.uncertain_prob_high,
    )
    payload['model_used'] = 'hybrid'
    payload['calibrated'] = hybrid_cal_applied
    payload['inference_backend'] = 'hybrid'
    payload['raw_probabilities'] = {
        'real': float(raw_prob_real_hybrid),
        'fake': float(1.0 - raw_prob_real_hybrid),
    }
    payload['component_probabilities'] = {
        'cnn_real': prob_real_cnn,
        'hybrid_real': prob_real_hybrid,
    }
    payload['preprocessing'] = pre_meta
    return payload


def load_image_from_path(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise FileNotFoundError(f'Image file does not exist: {image_path}')
    try:
        with Image.open(image_path) as img:
            return img.convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f'Invalid or corrupted image file: {image_path}') from exc


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('Uploaded file is not a valid image.') from exc
