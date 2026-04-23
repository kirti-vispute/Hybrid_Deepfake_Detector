from __future__ import annotations

import logging
from dataclasses import asdict
from http import HTTPStatus

from flask import Blueprint, current_app, jsonify, request

from backend.services.prediction_service import PredictionService, PredictionServiceError
from backend.utils.json_sanitize import sanitize_for_json

api_bp = Blueprint('api', __name__, url_prefix='/api')
LOGGER = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_MIME_PREFIX = 'image/'


def _json_error(message: str, error_code: str, status_code: int, details: str | None = None):
    payload = {
        'error_code': error_code,
        'message': message,
    }
    if details:
        payload['details'] = details
    return jsonify(payload), status_code


def _service() -> PredictionService:
    return current_app.config['prediction_service']


@api_bp.get('/health')
def health():
    availability = _service().get_model_availability()
    return jsonify(
        {
            'status': 'ok',
            'service': 'hybrid-deepfake-detector-backend',
            'models': asdict(availability),
        }
    )


@api_bp.get('/models')
def models():
    availability = _service().get_model_availability()
    hybrid_servable = bool(availability.hybrid or availability.classical)
    return jsonify(
        {
            'available': {'hybrid': hybrid_servable, 'classical': availability.classical},
            'default_model': 'classical' if availability.classical else 'hybrid',
            'public_modes': ['hybrid', 'classical'],
        }
    )


@api_bp.post('/predict')
def predict():
    if 'file' not in request.files:
        return _json_error('No file part in request.', 'missing_file', HTTPStatus.BAD_REQUEST)

    model_choice = request.form.get('model', 'hybrid').lower().strip()
    if model_choice in {'cnn', 'smart'}:
        model_choice = 'hybrid'
    file = request.files['file']
    reference_file = request.files.get('reference_file')

    if not file or not file.filename:
        return _json_error('No file selected.', 'empty_file', HTTPStatus.BAD_REQUEST)

    suffix = ''
    if '.' in file.filename:
        suffix = file.filename[file.filename.rfind('.'):].lower()

    if suffix not in ALLOWED_EXTENSIONS:
        return _json_error(
            f'Unsupported file type: {suffix or "unknown"}.',
            'unsupported_file_type',
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )

    content_type = (file.mimetype or '').lower()
    if content_type and not content_type.startswith(ALLOWED_MIME_PREFIX):
        return _json_error(
            f'Invalid MIME type: {content_type}.',
            'invalid_mime_type',
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )

    image_bytes = file.read()
    if not image_bytes:
        return _json_error('Uploaded file is empty.', 'empty_upload', HTTPStatus.BAD_REQUEST)

    max_size = current_app.config['MAX_CONTENT_LENGTH']
    if max_size and len(image_bytes) > max_size:
        return _json_error(
            'Uploaded file exceeds the configured size limit.',
            'file_too_large',
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    try:
        LOGGER.info(
            'POST /api/predict model=%s bytes=%s',
            model_choice,
            len(image_bytes),
        )
        reference_image_bytes = None
        if reference_file and reference_file.filename:
            reference_image_bytes = reference_file.read()
            if not reference_image_bytes:
                return _json_error('Reference file is empty.', 'empty_reference_upload', HTTPStatus.BAD_REQUEST)
        result = _service().predict(
            image_bytes=image_bytes,
            model_choice=model_choice,
            reference_image_bytes=reference_image_bytes,
        )
        safe = sanitize_for_json(result)
        return jsonify(safe)
    except PredictionServiceError as exc:
        LOGGER.warning(
            'Prediction rejected: code=%s message=%s details=%s',
            exc.error_code,
            exc.message,
            exc.details,
        )
        return _json_error(
            message=exc.message,
            error_code=exc.error_code,
            status_code=HTTPStatus.BAD_REQUEST,
            details=exc.details,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception('Unexpected error in /api/predict')
        return _json_error(
            message='Inference failed due to an unexpected server error.',
            error_code='predict_endpoint_error',
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            details=str(exc),
        )
