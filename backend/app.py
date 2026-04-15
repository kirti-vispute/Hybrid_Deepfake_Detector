from __future__ import annotations

import logging
import os
import sys
from http import HTTPStatus
from pathlib import Path

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import api_bp
from backend.services.prediction_service import PredictionService
from utils.config import get_config


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s - %(message)s',
)
LOGGER = logging.getLogger(__name__)


def create_app() -> Flask:
    config = get_config()

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = config.max_upload_size_bytes
    app.config['prediction_service'] = PredictionService(config)

    CORS(app, resources={r'/api/*': {'origins': '*'}})
    app.register_blueprint(api_bp)

    run_startup_validation = os.getenv('HDFD_VALIDATE_DATASET_ON_STARTUP', '0') == '1'
    if run_startup_validation:
        try:
            # Lazy import avoids pulling DL stack during classical-fallback startup.
            from utils.data_loader import validate_dataset
            summary = validate_dataset(config, strict_missing_ratio=config.max_missing_ratio)
            LOGGER.info('Dataset validation summary: %s', summary)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('Dataset validation warning at startup: %s', exc)
    else:
        LOGGER.info('Skipping full dataset validation on backend startup (set HDFD_VALIDATE_DATASET_ON_STARTUP=1 to enable).')

    app.config['prediction_service'].log_artifact_status()
    LOGGER.info('Registered API blueprint at url_prefix=/api (routes: /health, /models, /predict).')

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_error):
        return (
            jsonify(
                {
                    'error_code': 'file_too_large',
                    'message': 'File is too large.',
                    'details': f'Max upload size is {config.max_upload_size_mb} MB.',
                }
            ),
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return (
            jsonify(
                {
                    'error_code': 'http_error',
                    'message': error.description,
                    'details': error.name,
                }
            ),
            error.code or HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        LOGGER.exception('Unhandled backend exception: %s', error)
        payload = {
            'error_code': 'internal_server_error',
            'message': 'Unexpected server error.',
        }
        if os.getenv('HDFD_EXPOSE_ERROR_DETAILS', '').strip().lower() in {'1', 'true', 'yes'}:
            payload['details'] = str(error)
        return jsonify(payload), HTTPStatus.INTERNAL_SERVER_ERROR

    return app


app = create_app()


if __name__ == '__main__':
    cfg = get_config()
    app.run(host='0.0.0.0', port=cfg.backend_port, debug=False)
