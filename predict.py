from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

from utils.config import get_config
from utils.inference_utils import load_image_from_path, predict_pairwise_identity_aware, predict_pil_image
from utils.model_utils import load_cnn_model, load_xgb_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('predict')

BACKBONE_CHOICES = ['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Run local deepfake prediction from an image.')
    parser.add_argument('--image', required=True, type=str, help='Path to image file')
    parser.add_argument('--model', default='hybrid', choices=['cnn', 'hybrid', 'smart'], help='Public mode is hybrid; cnn/smart map to hybrid for compatibility.')
    parser.add_argument('--cnn-model', default=str(cfg.cnn_model_path), type=str)
    parser.add_argument('--xgb-model', default=str(cfg.xgb_model_path), type=str)
    parser.add_argument('--reference-image', default=None, type=str, help='Optional trusted real reference image of the same person.')
    parser.add_argument('--image-size', default=cfg.image_size, type=int)
    parser.add_argument('--model-type', default=cfg.backbone_name, choices=BACKBONE_CHOICES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    image_path = Path(args.image).resolve()

    try:
        image = load_image_from_path(image_path)
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1) from exc
    except ValueError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1) from exc

    reference_image = None
    if args.reference_image:
        try:
            reference_image = load_image_from_path(Path(args.reference_image).resolve())
        except (FileNotFoundError, ValueError) as exc:
            LOGGER.error('Invalid reference image: %s', exc)
            raise SystemExit(1) from exc

    cfg.image_size = args.image_size
    cfg.model_type = args.model_type
    cfg.backbone_name = args.model_type

    try:
        cnn_model = load_cnn_model(Path(args.cnn_model).resolve())
        xgb_model = None
        xgb_model = load_xgb_model(Path(args.xgb_model).resolve())

        if reference_image is not None:
            result = predict_pairwise_identity_aware(
                reference_image=reference_image,
                candidate_image=image,
                config=cfg,
                cnn_model=cnn_model,
                xgb_model=xgb_model,
            )
        else:
            result = predict_pil_image(
                image=image,
                model_choice=args.model,
                config=cfg,
                cnn_model=cnn_model,
                xgb_model=xgb_model,
            )
    except FileNotFoundError as exc:
        LOGGER.error('Missing model file: %s', exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        LOGGER.error('Invalid input: %s', exc)
        raise SystemExit(1) from exc
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error('Prediction failed: %s', exc)
        raise SystemExit(1) from exc

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
