from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

import pandas as pd

from utils.config import get_config
from utils.data_loader import audit_dataset, load_split_dataframe
from utils.inference_utils import load_image_from_path, predict_pil_image
from utils.model_utils import load_cnn_model, load_optional_joblib, load_xgb_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('sanity_checks')


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Run strict sanity checks for dataset paths, labels, and optional model predictions.')
    parser.add_argument('--output-json', type=str, default=str(cfg.results_dir / 'sanity_check_report.json'))
    parser.add_argument('--max-samples-per-split', type=int, default=5)
    parser.add_argument('--check-corrupt', action='store_true')
    parser.add_argument('--max-corrupt-checks-per-split', type=int, default=200)
    parser.add_argument('--with-model-check', action='store_true')
    parser.add_argument('--prediction-samples', type=int, default=8)
    return parser.parse_args()


def _sample_predictions(num_samples: int) -> list[dict]:
    cfg = get_config()
    train_split = load_split_dataframe('train', cfg, max_samples=max_samples_for_prediction(num_samples))
    df: pd.DataFrame = train_split.dataframe.head(num_samples)

    cnn_model = load_cnn_model(cfg.cnn_model_path)
    xgb_model = load_xgb_model(cfg.xgb_model_path) if cfg.xgb_model_path.exists() else None
    scaler = load_optional_joblib(cfg.xgb_scaler_path)

    rows: list[dict] = []
    for _, row in df.iterrows():
        image = load_image_from_path(Path(row['abs_path']))
        entry = {
            'path': str(row['path']),
            'abs_path': str(row['abs_path']),
            'label': int(row['label']),
            'label_str': str(row['label_str']),
        }
        cnn_pred = predict_pil_image(image=image, model_choice='cnn', config=cfg, cnn_model=cnn_model)
        entry['cnn'] = cnn_pred

        if xgb_model is not None:
            hybrid_pred = predict_pil_image(
                image=image,
                model_choice='hybrid',
                config=cfg,
                cnn_model=cnn_model,
                xgb_model=xgb_model,
                feature_scaler=scaler,
            )
            entry['hybrid'] = hybrid_pred

        rows.append(entry)
    return rows


def max_samples_for_prediction(num_samples: int) -> int:
    return max(64, num_samples * 4)


def main() -> None:
    args = parse_args()
    cfg = get_config()

    report = audit_dataset(
        config=cfg,
        check_corrupt=args.check_corrupt,
        max_corrupt_checks_per_split=args.max_corrupt_checks_per_split,
        max_log_samples=args.max_samples_per_split,
        output_json=None,
    )

    if args.with_model_check:
        report['prediction_sanity'] = _sample_predictions(args.prediction_samples)

    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2)

    print(json.dumps(report, indent=2))
    LOGGER.info('Sanity report saved to: %s', output)


if __name__ == '__main__':
    main()
