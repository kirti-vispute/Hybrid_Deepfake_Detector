from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock

from PIL import Image

from utils.config import AppConfig
from utils.classical_inference import load_classical_bundle, predict_classical_from_bytes, predict_classical_from_pil

LOGGER = logging.getLogger(__name__)


class PredictionServiceError(Exception):
    def __init__(self, message: str, error_code: str = 'prediction_error', details: str | None = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details


@dataclass
class ModelAvailability:
    cnn: bool
    hybrid: bool
    smart: bool
    classical: bool


def _production_is_cnn_direct(config: AppConfig) -> bool:
    payload = _load_optional_json(config.production_inference_path) or {}
    backend = str(payload.get('backend', 'hybrid')).lower().strip()
    return backend in {'cnn', 'cnn_direct'}


def _production_is_classical(config: AppConfig) -> bool:
    payload = _load_optional_json(config.production_inference_path) or {}
    backend = str(payload.get('backend', 'hybrid')).lower().strip()
    return backend in {'classical', 'classical_fallback', 'fallback'}


def _cnn_metadata_passes_collapse_guard(config: AppConfig) -> bool:
    payload = _load_optional_json(config.cnn_metadata_path) or {}
    guard = payload.get('collapse_guard')
    if not isinstance(guard, dict):
        return True
    return bool(guard.get('passed', True))


def _classical_metadata_passes_collapse_guard(config: AppConfig) -> bool:
    payload = _load_optional_json(config.classical_metadata_path) or {}
    guard = payload.get('collapse_guard')
    if not isinstance(guard, dict):
        return True
    return bool(guard.get('passed', True))


def _load_optional_json(path):
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return None


def _decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            return img.convert('RGB')
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError('Uploaded file is not a valid image.') from exc


class PredictionService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = Lock()
        self._cnn_model = None
        self._xgb_model = None
        self._xgb_scaler = None
        self._cnn_calibrator = None
        self._hybrid_calibrator = None
        self._classical_bundle = None
        self._classical_bundle_signature = None

    def _classical_artifact_signature(self) -> tuple[int | None, int | None]:
        def _mtime_ns(path: Path) -> int | None:
            if not path.exists():
                return None
            try:
                return int(path.stat().st_mtime_ns)
            except OSError:
                return None

        return (
            _mtime_ns(self.config.classical_model_path),
            _mtime_ns(self.config.classical_metadata_path),
        )

    def get_model_availability(self) -> ModelAvailability:
        cnn_artifact_exists = self.config.cnn_model_path.exists()
        cnn_guard_ok = _cnn_metadata_passes_collapse_guard(self.config)
        cnn_available = bool(cnn_artifact_exists and cnn_guard_ok)
        xgb_available = self.config.xgb_model_path.exists()
        classical_artifacts_exist = self.config.classical_model_path.exists() and self.config.classical_metadata_path.exists()
        classical_guard_ok = _classical_metadata_passes_collapse_guard(self.config)
        classical_available = bool(classical_artifacts_exist and classical_guard_ok)
        use_cnn_only = _production_is_cnn_direct(self.config)
        hybrid_available = (cnn_available and (use_cnn_only or xgb_available)) or classical_available
        smart_payload = _load_optional_json(self.config.smart_router_path) if self.config.smart_router_path.exists() else None
        smart_available = hybrid_available and bool((smart_payload or {}).get('enabled', False))
        return ModelAvailability(cnn=cnn_available, hybrid=hybrid_available, smart=smart_available, classical=classical_available)

    def log_artifact_status(self) -> None:
        """Log which artifact files exist (paths resolved from config)."""
        cfg = self.config
        rows = [
            ('cnn_model', cfg.cnn_model_path),
            ('xgb_model', cfg.xgb_model_path),
            ('xgb_scaler', cfg.xgb_scaler_path),
            ('cnn_calibrator', cfg.cnn_calibrator_path),
            ('hybrid_calibrator', cfg.hybrid_calibrator_path),
            ('cnn_metadata', cfg.cnn_metadata_path),
            ('hybrid_metadata', cfg.hybrid_metadata_path),
            ('classical_model', cfg.classical_model_path),
            ('classical_metadata', cfg.classical_metadata_path),
            ('production_inference', cfg.production_inference_path),
        ]
        for name, path in rows:
            status = 'present' if path.exists() else 'MISSING'
            LOGGER.info('Artifact %-22s %s | %s', name, status, path)
        if cfg.cnn_model_path.exists() and not _cnn_metadata_passes_collapse_guard(cfg):
            LOGGER.warning(
                'CNN artifact exists but collapse guard failed in %s; excluding CNN from active availability.',
                cfg.cnn_metadata_path,
            )
        if cfg.classical_model_path.exists() and cfg.classical_metadata_path.exists() and not _classical_metadata_passes_collapse_guard(cfg):
            LOGGER.warning(
                'Classical artifact exists but collapse guard failed in %s; excluding classical model.',
                cfg.classical_metadata_path,
            )
        avail = self.get_model_availability()
        LOGGER.info(
            'Model availability: cnn=%s hybrid=%s smart=%s classical=%s (production_backend=%s)',
            avail.cnn,
            avail.hybrid,
            avail.smart,
            avail.classical,
            'cnn_direct' if _production_is_cnn_direct(cfg) else ('classical_fallback' if _production_is_classical(cfg) else 'hybrid'),
        )

    def _get_classical_bundle(self):
        current_signature = self._classical_artifact_signature()
        if self._classical_bundle is None or self._classical_bundle_signature != current_signature:
            try:
                self._classical_bundle = load_classical_bundle(self.config)
                self._classical_bundle_signature = current_signature
            except Exception as exc:  # pylint: disable=broad-except
                raise PredictionServiceError(
                    message='Classical fallback model is unavailable.',
                    error_code='missing_classical_model',
                    details=str(exc),
                ) from exc
        return self._classical_bundle

    def predict(self, image_bytes: bytes, model_choice: str, reference_image_bytes: bytes | None = None) -> dict:
        model_choice = model_choice.lower().strip()
        if model_choice in {'cnn', 'smart'}:
            model_choice = 'hybrid'
        if model_choice not in {'hybrid', 'classical', 'fallback'}:
            raise PredictionServiceError(
                message="Invalid model selection. Use 'hybrid' or 'classical'.",
                error_code='invalid_model_choice',
            )
        if model_choice == 'fallback':
            model_choice = 'classical'

        # Decode image ONCE here — all downstream paths reuse this PIL object.
        try:
            image = _decode_image_bytes(image_bytes)
            assert isinstance(image, Image.Image)
            LOGGER.info('Image decoded successfully (%sx%s RGB).', image.width, image.height)
        except ValueError as exc:
            raise PredictionServiceError(
                message='Uploaded file is not a valid image.',
                error_code='invalid_image',
                details=str(exc),
            ) from exc

        reference_image = None
        if reference_image_bytes:
            try:
                reference_image = _decode_image_bytes(reference_image_bytes)
            except ValueError as exc:
                raise PredictionServiceError(
                    message='Reference file is not a valid image.',
                    error_code='invalid_reference_image',
                    details=str(exc),
                ) from exc

        try:
            if model_choice == 'classical':
                # Availability check is stateless — no lock needed.
                availability = self.get_model_availability()
                if not availability.classical:
                    raise PredictionServiceError(
                        message='Classical model failed safety checks and is disabled.',
                        error_code='classical_model_rejected',
                    )
                # Lock scope: only covers the lazy bundle load, not feature extraction.
                with self._lock:
                    bundle = self._get_classical_bundle()
                # Inference runs outside the lock — stateless and thread-safe.
                result = predict_classical_from_pil(pil_image=image, config=self.config, bundle=bundle)

            else:
                availability = self.get_model_availability()
                if not (availability.cnn and (_production_is_cnn_direct(self.config) or self.config.xgb_model_path.exists())):
                    if availability.classical:
                        with self._lock:
                            bundle = self._get_classical_bundle()
                        result = predict_classical_from_pil(pil_image=image, config=self.config, bundle=bundle)
                        result['requested_model'] = 'hybrid'
                        result['fallback_reason'] = 'hybrid_artifacts_unavailable'
                        LOGGER.info('Hybrid unavailable; served classical fallback for hybrid request.')
                        return result

                # Lazy import to avoid loading DL stack when using classical fallback.
                from utils.calibration_utils import load_optional_calibrator
                from utils.inference_utils import predict_pairwise_identity_aware, predict_pil_image
                from utils.model_utils import load_cnn_model, load_optional_joblib, load_xgb_model

                # Lock only for lazy model loads; capture references then release.
                with self._lock:
                    if self._cnn_model is None:
                        self._cnn_model = load_cnn_model(self.config.cnn_model_path)
                        self._cnn_calibrator = load_optional_calibrator(self.config.cnn_calibrator_path)
                    if self._xgb_model is None and not _production_is_cnn_direct(self.config):
                        self._xgb_model = load_xgb_model(self.config.xgb_model_path)
                        self._xgb_scaler = load_optional_joblib(self.config.xgb_scaler_path)
                        self._hybrid_calibrator = load_optional_calibrator(self.config.hybrid_calibrator_path)
                    # Snapshot references while holding lock so inference runs outside.
                    cnn_model = self._cnn_model
                    xgb_model = self._xgb_model
                    xgb_scaler = self._xgb_scaler
                    cnn_calibrator = self._cnn_calibrator
                    hybrid_calibrator = self._hybrid_calibrator

                if reference_image is not None:
                    result = predict_pairwise_identity_aware(
                        reference_image=reference_image,
                        candidate_image=image,
                        config=self.config,
                        cnn_model=cnn_model,
                        xgb_model=xgb_model,
                        feature_scaler=xgb_scaler,
                        cnn_calibrator=cnn_calibrator,
                        hybrid_calibrator=hybrid_calibrator,
                    )
                else:
                    result = predict_pil_image(
                        image=image,
                        model_choice='hybrid',
                        config=self.config,
                        cnn_model=cnn_model,
                        xgb_model=xgb_model,
                        feature_scaler=xgb_scaler,
                        cnn_calibrator=cnn_calibrator,
                        hybrid_calibrator=hybrid_calibrator,
                    )

            LOGGER.info(
                'Prediction OK: class=%s confidence=%.4f backend=%s',
                result.get('predicted_class'),
                float(result.get('confidence', 0.0)),
                result.get('inference_backend'),
            )
            return result

        except PredictionServiceError:
            raise
        except FileNotFoundError as exc:
            LOGGER.exception('Missing file during prediction')
            raise PredictionServiceError(
                message='A required model file is missing.',
                error_code='missing_model_file',
                details=str(exc),
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception('Prediction pipeline failed')
            raise PredictionServiceError(
                message='Prediction failed due to an internal processing error.',
                error_code='prediction_runtime_error',
                details=str(exc),
            ) from exc
