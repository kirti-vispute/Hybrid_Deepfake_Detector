from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging

import numpy as np

from utils.calibration_utils import apply_calibration, load_optional_calibrator
from utils.config import get_config
from utils.data_loader import create_split_sequence, load_split_dataframe
from utils.fusion_utils import ClassAwareFusionParams, apply_class_aware_fusion
from utils.metrics_utils import compute_classification_metrics, save_confusion_matrix_plot, save_json
from utils.model_utils import (
    get_feature_extractor,
    get_model_input_size,
    load_cnn_model,
    load_joblib,
    load_optional_joblib,
    load_optional_json,
    load_xgb_model,
    normalize_model_type,
    predict_cnn_real_probs,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('evaluate')

BACKBONE_CHOICES = ['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Evaluate CNN, Hybrid, and optional Smart fusion on splits.')
    parser.add_argument('--cnn-model', type=str, default=str(cfg.cnn_model_path))
    parser.add_argument('--xgb-model', type=str, default=str(cfg.xgb_model_path))
    parser.add_argument('--feature-dir', type=str, default=str(cfg.feature_dir))
    parser.add_argument('--results-dir', type=str, default=str(cfg.results_dir))
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--model-type', type=str, default=cfg.backbone_name, choices=BACKBONE_CHOICES)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-valid-samples', type=int, default=None)
    parser.add_argument('--max-test-samples', type=int, default=None)
    parser.add_argument('--skip-train-valid', action='store_true')
    parser.add_argument('--disable-smart', action='store_true')
    return parser.parse_args()


def _load_features(feature_dir: Path, split: str):
    path = feature_dir / f'{split}_features.npz'
    if not path.exists():
        return None, None
    data = np.load(path)
    return data['features'], data['labels']


def _subset_arrays(features: np.ndarray, labels: np.ndarray, max_samples: int | None):
    if max_samples is None or max_samples <= 0 or max_samples >= len(labels):
        return features, labels
    idx = np.arange(len(labels))[:max_samples]
    return features[idx], labels[idx]


def _threshold_from_metadata(path: Path, default: float = 0.5) -> float:
    metadata = load_optional_json(path) or {}
    try:
        return float(metadata.get('decision_threshold', default))
    except (TypeError, ValueError):
        return float(default)


def _smart_params_from_router(router_payload: dict | None) -> ClassAwareFusionParams | None:
    if not router_payload or not bool(router_payload.get('enabled', False)):
        return None
    params = router_payload.get('params', {})
    required = {'cnn_weight', 'real_gate', 'fake_gate', 'decision_threshold'}
    if not required.issubset(set(params.keys())):
        return None
    return ClassAwareFusionParams(
        cnn_weight=float(params['cnn_weight']),
        real_gate=float(params['real_gate']),
        fake_gate=float(params['fake_gate']),
        decision_threshold=float(params['decision_threshold']),
    )


def _macro_f1_from_split_metrics(calibrated_block: dict) -> float:
    per_class = calibrated_block.get('per_class') or {}
    real_f1 = float((per_class.get('real') or {}).get('f1_score', 0.0))
    fake_f1 = float((per_class.get('fake') or {}).get('f1_score', 0.0))
    return (real_f1 + fake_f1) / 2.0


def _write_production_inference_choice(cfg, report: dict) -> None:
    if 'test' not in report:
        return
    test_block = report['test']
    cnn_cal = test_block.get('cnn', {}).get('calibrated')
    hyb_cal = test_block.get('hybrid', {}).get('calibrated')
    if not cnn_cal or not hyb_cal:
        return
    test_rows = int(test_block.get('rows') or 0)
    cnn_macro = _macro_f1_from_split_metrics(cnn_cal)
    hyb_macro = _macro_f1_from_split_metrics(hyb_cal)
    margin = 0.005
    # Small test splits make macro-F1 comparisons noisy; default to hybrid unless CNN is clearly ahead on enough data.
    if test_rows < 200:
        backend = 'hybrid'
        reason = 'test_set_small_default_hybrid'
    elif cnn_macro > hyb_macro + margin:
        backend = 'cnn_direct'
        reason = 'test_macro_f1_cnn_ahead'
    else:
        backend = 'hybrid'
        reason = 'test_macro_f1_hybrid_ahead_or_tie'
    payload = {
        'backend': backend,
        'reason': reason,
        'test_cnn_macro_f1': cnn_macro,
        'test_hybrid_macro_f1': hyb_macro,
        'margin': margin,
    }
    save_json(payload, cfg.production_inference_path)
    LOGGER.info('Production inference routing saved to %s: %s', cfg.production_inference_path, payload)


def _evaluate_split(
    split: str,
    cfg,
    cnn_model,
    xgb_model,
    hybrid_scaler,
    hybrid_pca,
    use_platt_hybrid: bool,
    cnn_calibrator,
    hybrid_calibrator,
    feature_dir: Path,
    model_type: str,
    batch_size: int,
    image_size: int,
    cnn_threshold: float,
    hybrid_threshold: float,
    smart_params: ClassAwareFusionParams | None,
    max_samples: int | None,
    dual_hybrid_rgb: bool = False,
    class_rgb_stats: dict | None = None,
):
    split_data = load_split_dataframe(split, cfg, max_samples=max_samples)
    split_seq = create_split_sequence(
        split_data=split_data,
        config=cfg,
        model_type=model_type,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=True,
        rgb_input_only=dual_hybrid_rgb,
        auxiliary_quality=False,
        class_rgb_stats=class_rgb_stats,
    )

    y_true = split_seq.get_labels()

    cnn_probs_raw = predict_cnn_real_probs(cnn_model, split_seq, verbose=0)
    cnn_probs_cal, _ = apply_calibration(cnn_probs_raw, cnn_calibrator)

    cnn_metrics_raw = compute_classification_metrics(y_true=y_true, y_prob=cnn_probs_raw)
    cnn_metrics_cal = compute_classification_metrics(y_true=y_true, y_prob=cnn_probs_cal, threshold=cnn_threshold)

    split_features, feature_labels = _load_features(feature_dir, split)
    if split_features is None:
        extractor = get_feature_extractor(cnn_model)
        split_features = extractor.predict(split_seq, verbose=0)
        feature_labels = y_true
    else:
        split_features, feature_labels = _subset_arrays(split_features, feature_labels, max_samples)

    if feature_labels is not None and not np.array_equal(feature_labels.astype(int), y_true.astype(int)):
        LOGGER.warning('[%s] Label mismatch between sequence and saved features; using sequence labels.', split)

    if hybrid_scaler is not None:
        split_features = hybrid_scaler.transform(split_features)
    if hybrid_pca is not None:
        split_features = hybrid_pca.transform(split_features)

    hybrid_probs_raw = xgb_model.predict_proba(split_features)[:, 1]
    if use_platt_hybrid:
        hybrid_probs_cal, _ = apply_calibration(hybrid_probs_raw, hybrid_calibrator)
    else:
        hybrid_probs_cal = hybrid_probs_raw

    hybrid_metrics_raw = compute_classification_metrics(y_true=y_true, y_prob=hybrid_probs_raw)
    hybrid_metrics_cal = compute_classification_metrics(y_true=y_true, y_prob=hybrid_probs_cal, threshold=hybrid_threshold)

    smart_block = None
    if smart_params is not None:
        smart_probs = apply_class_aware_fusion(cnn_prob=cnn_probs_cal, hybrid_prob=hybrid_probs_cal, params=smart_params)
        smart_metrics = compute_classification_metrics(
            y_true=y_true,
            y_prob=smart_probs,
            threshold=smart_params.decision_threshold,
        )
        smart_block = {
            'calibrated': smart_metrics,
            'threshold': float(smart_params.decision_threshold),
            'params': {
                'cnn_weight': smart_params.cnn_weight,
                'real_gate': smart_params.real_gate,
                'fake_gate': smart_params.fake_gate,
            },
        }

    return {
        'rows': int(len(y_true)),
        'cnn': {
            'raw': cnn_metrics_raw,
            'calibrated': cnn_metrics_cal,
            'threshold': float(cnn_threshold),
        },
        'hybrid': {
            'raw': hybrid_metrics_raw,
            'calibrated': hybrid_metrics_cal,
            'threshold': float(hybrid_threshold),
        },
        'smart': smart_block,
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    cnn_model = load_cnn_model(Path(args.cnn_model).resolve())
    cnn_metadata = load_optional_json(cfg.cnn_metadata_path) or {}
    model_type = normalize_model_type(str(cnn_metadata.get('backbone_name', cnn_metadata.get('model_type', args.model_type))))
    image_size = int(cnn_metadata.get('image_size', get_model_input_size(cnn_model)))
    dual_hybrid_rgb = bool(cnn_metadata.get('dual_hybrid_backbone', False))
    class_rgb_stats = None
    stats_path = Path(cnn_metadata.get('class_rgb_stats_path', cfg.cnn_train_rgb_stats_path))
    if dual_hybrid_rgb and stats_path.exists():
        try:
            class_rgb_stats = load_joblib(stats_path)
        except OSError:
            class_rgb_stats = None

    xgb_model = load_xgb_model(Path(args.xgb_model).resolve())
    hybrid_scaler = load_optional_joblib(cfg.xgb_scaler_path)
    cnn_calibrator = load_optional_calibrator(cfg.cnn_calibrator_path)
    hybrid_calibrator = load_optional_calibrator(cfg.hybrid_calibrator_path)

    cnn_threshold = _threshold_from_metadata(cfg.cnn_metadata_path, default=0.5)
    hybrid_meta = load_optional_json(cfg.hybrid_metadata_path) or {}
    hybrid_threshold = float(hybrid_meta.get('decision_threshold', _threshold_from_metadata(cfg.hybrid_metadata_path, default=0.5)))
    use_platt_hybrid = bool(hybrid_meta.get('use_platt_overlay', True))
    pca_path = hybrid_meta.get('pca_path')
    hybrid_pca = load_optional_joblib(Path(pca_path)) if pca_path else load_optional_joblib(cfg.hybrid_pca_path)

    smart_router_payload = None if args.disable_smart else load_optional_json(cfg.smart_router_path)
    smart_params = _smart_params_from_router(smart_router_payload)

    feature_dir = Path(args.feature_dir).resolve()

    split_limits = {
        'train': args.max_train_samples,
        'valid': args.max_valid_samples,
        'test': args.max_test_samples,
    }

    splits = ['test'] if args.skip_train_valid else ['train', 'valid', 'test']

    report = {}
    for split in splits:
        LOGGER.info('Evaluating split=%s', split)
        report[split] = _evaluate_split(
            split=split,
            cfg=cfg,
            cnn_model=cnn_model,
            xgb_model=xgb_model,
            hybrid_scaler=hybrid_scaler,
            hybrid_pca=hybrid_pca,
            use_platt_hybrid=use_platt_hybrid,
            cnn_calibrator=cnn_calibrator,
            hybrid_calibrator=hybrid_calibrator,
            feature_dir=feature_dir,
            model_type=model_type,
            batch_size=args.batch_size,
            image_size=image_size,
            cnn_threshold=cnn_threshold,
            hybrid_threshold=hybrid_threshold,
            smart_params=smart_params,
            max_samples=split_limits[split],
            dual_hybrid_rgb=dual_hybrid_rgb,
            class_rgb_stats=class_rgb_stats,
        )

    report['meta'] = {
        'model_type': model_type,
        'image_size': image_size,
        'dual_hybrid_rgb_input': dual_hybrid_rgb,
        'cnn_threshold': cnn_threshold,
        'hybrid_threshold': hybrid_threshold,
        'hybrid_pca_loaded': hybrid_pca is not None,
        'hybrid_use_platt_overlay': use_platt_hybrid,
        'smart_enabled': smart_params is not None,
    }

    save_json(report, results_dir / 'evaluation_comparison.json')
    _write_production_inference_choice(cfg, report)

    if 'test' in report:
        cnn_cm = np.array(report['test']['cnn']['calibrated']['confusion_matrix'])
        hybrid_cm = np.array(report['test']['hybrid']['calibrated']['confusion_matrix'])

        save_confusion_matrix_plot(cnn_cm, ['Fake', 'Real'], 'CNN Baseline (Calibrated) Test Confusion Matrix', results_dir / 'cnn_confusion_matrix.png')
        save_confusion_matrix_plot(
            hybrid_cm,
            ['Fake', 'Real'],
            'Hybrid CNN + XGBoost (Calibrated) Test Confusion Matrix',
            results_dir / 'hybrid_confusion_matrix.png',
        )

        if report['test'].get('smart'):
            smart_cm = np.array(report['test']['smart']['calibrated']['confusion_matrix'])
            save_confusion_matrix_plot(
                smart_cm,
                ['Fake', 'Real'],
                'Smart Class-Aware Fusion (Calibrated) Test Confusion Matrix',
                results_dir / 'smart_confusion_matrix.png',
            )

    LOGGER.info('Saved evaluation report to %s', results_dir / 'evaluation_comparison.json')


if __name__ == '__main__':
    main()
