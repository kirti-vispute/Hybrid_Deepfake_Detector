from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.classical_features import extract_identity_features_from_pil
from utils.calibration_utils import ProbabilityCalibrator, apply_calibration, load_optional_calibrator
from utils.config import AppConfig
from utils.fusion_utils import ClassAwareFusionParams, apply_class_aware_fusion
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
    image: Image.Image,
    cnn_model,
    xgb_model,
    config: AppConfig,
    feature_scaler=None,
    pca=None,
) -> float:
    extractor = get_feature_extractor(cnn_model)
    features = extractor.predict(batch, verbose=0)
    identity_features = extract_identity_features_from_pil(image=image, face_crop_expand=config.face_crop_expand)
    features = np.concatenate([features.astype(np.float32), identity_features.reshape(1, -1)], axis=1)
    if feature_scaler is not None:
        features = feature_scaler.transform(features)
    if pca is not None:
        features = pca.transform(features)
    probabilities = xgb_model.predict_proba(features)
    return float(probabilities[:, 1][0])


def _embedding_for_image(image: Image.Image, config: AppConfig, cnn_model) -> np.ndarray:
    model_type, image_size = _resolve_model_runtime_settings(config=config, cnn_model=cnn_model)
    batch, _meta = preprocess_single_image_with_meta(image, image_size, model_type)
    extractor = get_feature_extractor(cnn_model)
    emb = extractor.predict(batch, verbose=0)
    return np.asarray(emb, dtype=np.float32).reshape(-1)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    return float(np.dot(a, b) / denom)


def _pair_feature_vector(
    reference_image: Image.Image,
    candidate_image: Image.Image,
    config: AppConfig,
    cnn_model,
) -> np.ndarray:
    ref_emb = _embedding_for_image(reference_image, config=config, cnn_model=cnn_model)
    cand_emb = _embedding_for_image(candidate_image, config=config, cnn_model=cnn_model)
    ref_id = extract_identity_features_from_pil(reference_image, face_crop_expand=config.face_crop_expand)
    cand_id = extract_identity_features_from_pil(candidate_image, face_crop_expand=config.face_crop_expand)

    artifact_slice = slice(146, 152)
    symmetry_slice = slice(152, 155)
    faceprint_slice = slice(155, None)

    feats = np.array(
        [
            _cosine_similarity(ref_emb, cand_emb),
            float(np.linalg.norm(ref_emb - cand_emb)),
            _cosine_similarity(ref_id, cand_id),
            float(np.linalg.norm(ref_id - cand_id)),
            float(np.mean(np.abs(ref_id[artifact_slice] - cand_id[artifact_slice]))),
            float(np.mean(np.abs(ref_id[symmetry_slice] - cand_id[symmetry_slice]))),
            float(np.mean(np.abs(ref_id[faceprint_slice] - cand_id[faceprint_slice]))),
        ],
        dtype=np.float32,
    )
    return feats.reshape(1, -1)


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


def _smart_router_params(config: AppConfig) -> ClassAwareFusionParams | None:
    payload = load_optional_json(config.smart_router_path) or {}
    if not bool(payload.get('enabled', False)):
        return None
    params = payload.get('params', {})
    required = {'cnn_weight', 'real_gate', 'fake_gate', 'decision_threshold'}
    if not required.issubset(set(params.keys())):
        return None
    return ClassAwareFusionParams(
        cnn_weight=float(params['cnn_weight']),
        real_gate=float(params['real_gate']),
        fake_gate=float(params['fake_gate']),
        decision_threshold=float(params['decision_threshold']),
    )


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
    pca = None
    if bool(hybrid_meta.get('pca_used', True)):
        pca_path = hybrid_meta.get('pca_path')
        pca = load_optional_joblib(Path(pca_path)) if pca_path else load_optional_joblib(config.hybrid_pca_path)

    raw_prob_real_hybrid = _predict_prob_hybrid(
        batch,
        image=image,
        cnn_model=cnn_model,
        xgb_model=xgb_model,
        config=config,
        feature_scaler=scaler,
        pca=pca,
    )
    if use_platt_hybrid:
        hybrid_cal = hybrid_calibrator if hybrid_calibrator is not None else load_optional_calibrator(config.hybrid_calibrator_path)
        hybrid_probs, hybrid_cal_applied = apply_calibration(np.array([raw_prob_real_hybrid]), hybrid_cal)
        prob_real_hybrid = float(hybrid_probs[0])
    else:
        prob_real_hybrid = float(raw_prob_real_hybrid)
        hybrid_cal_applied = False

    smart_params = _smart_router_params(config)
    final_prob = prob_real_hybrid
    final_threshold = hybrid_threshold
    model_used = 'hybrid'
    if smart_params is not None:
        final_prob = float(
            apply_class_aware_fusion(
                cnn_prob=np.array([prob_real_cnn], dtype=np.float32),
                hybrid_prob=np.array([prob_real_hybrid], dtype=np.float32),
                params=smart_params,
            )[0]
        )
        final_threshold = float(smart_params.decision_threshold)
        model_used = 'smart_hybrid'

    payload = probability_to_prediction(
        final_prob,
        threshold=final_threshold,
        uncertain_low=config.uncertain_prob_low,
        uncertain_high=config.uncertain_prob_high,
    )
    payload['model_used'] = model_used
    payload['calibrated'] = hybrid_cal_applied
    payload['inference_backend'] = model_used
    payload['raw_probabilities'] = {
        'real': float(raw_prob_real_hybrid),
        'fake': float(1.0 - raw_prob_real_hybrid),
    }
    payload['component_probabilities'] = {
        'cnn_real': prob_real_cnn,
        'hybrid_real': prob_real_hybrid,
        'final_real': final_prob,
    }
    payload['preprocessing'] = pre_meta
    return payload


def predict_pairwise_identity_aware(
    reference_image: Image.Image,
    candidate_image: Image.Image,
    config: AppConfig,
    cnn_model=None,
    xgb_model=None,
    feature_scaler=None,
    cnn_calibrator: ProbabilityCalibrator | None = None,
    hybrid_calibrator: ProbabilityCalibrator | None = None,
) -> dict:
    cnn_model = cnn_model or load_cnn_model(config.cnn_model_path)
    base = predict_pil_image(
        image=candidate_image,
        model_choice='hybrid',
        config=config,
        cnn_model=cnn_model,
        xgb_model=xgb_model,
        feature_scaler=feature_scaler,
        cnn_calibrator=cnn_calibrator,
        hybrid_calibrator=hybrid_calibrator,
    )

    if not config.reference_pair_model_path.exists():
        base['reference_mode'] = False
        base['reference_reason'] = 'pair_model_missing'
        return base

    pair_model = joblib.load(config.reference_pair_model_path)
    pair_features = _pair_feature_vector(reference_image, candidate_image, config=config, cnn_model=cnn_model)
    pair_fake_prob = float(pair_model.predict_proba(pair_features)[0, 1])
    pair_real_prob = float(1.0 - pair_fake_prob)

    base_real_prob = float(base['probabilities']['real'])
    final_real_prob = min(base_real_prob, pair_real_prob)
    final_fake_prob = 1.0 - final_real_prob
    threshold = float((load_optional_json(config.hybrid_metadata_path) or {}).get('decision_threshold', 0.5))
    final = probability_to_prediction(final_real_prob, threshold=threshold, uncertain_low=config.uncertain_prob_low, uncertain_high=config.uncertain_prob_high)
    final['model_used'] = 'hybrid_reference'
    final['inference_backend'] = 'hybrid_reference'
    final['calibrated'] = bool(base.get('calibrated', False))
    final['decision_threshold'] = threshold
    final['raw_probabilities'] = base.get('raw_probabilities', {})
    final['component_probabilities'] = {
        **dict(base.get('component_probabilities', {})),
        'pair_real': pair_real_prob,
        'pair_fake': pair_fake_prob,
        'final_real': final_real_prob,
    }
    final['preprocessing'] = base.get('preprocessing', {})
    final['reference_mode'] = True
    return final


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
