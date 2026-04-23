from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from utils.calibration_utils import apply_calibration, fit_hybrid_calibrator_isotonic, fit_platt_calibrator, load_optional_calibrator, save_calibrator
from utils.config import get_config
from utils.data_loader import CSVImageSequence
from utils.fusion_utils import ClassAwareFusionParams, apply_class_aware_fusion, fit_class_aware_fusion, optimize_threshold
from utils.metrics_utils import compute_classification_metrics, save_json
from utils.model_utils import (
    get_model_input_size,
    load_cnn_model,
    load_joblib,
    load_optional_json,
    normalize_model_type,
    predict_cnn_real_probs,
    save_joblib,
    save_json_file,
    save_xgb_model,
)
from utils.threshold_utils import balanced_tpr_tnr_threshold
from utils.hybrid_classifier import SoftVotingHybridEnsemble

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('train_xgboost')


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Train calibrated hybrid classifier on extracted CNN features.')
    parser.add_argument('--mode', choices=['fast', 'strong', 'rigorous'], default='strong')
    parser.add_argument('--feature-dir', type=str, default=str(cfg.feature_dir))
    parser.add_argument('--output-model', type=str, default=str(cfg.xgb_model_path))
    parser.add_argument('--scaler-path', type=str, default=str(cfg.xgb_scaler_path))
    parser.add_argument('--pca-path', type=str, default=str(cfg.hybrid_pca_path))
    parser.add_argument('--calibrator-path', type=str, default=str(cfg.hybrid_calibrator_path))
    parser.add_argument('--metadata-path', type=str, default=str(cfg.hybrid_metadata_path))
    parser.add_argument('--smart-router-path', type=str, default=str(cfg.smart_router_path))
    parser.add_argument('--metrics-path', type=str, default=str(cfg.results_dir / 'xgboost_valid_metrics.json'))
    parser.add_argument('--n-estimators', type=int, default=900)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-valid-samples', type=int, default=None)
    parser.add_argument('--disable-smart-router', action='store_true')
    parser.add_argument('--disable-svm', action='store_true')
    parser.add_argument('--disable-ensemble', action='store_true')
    parser.add_argument(
        '--legacy-ensemble',
        action='store_true',
        help='Use legacy grid search + optional SVM/ensemble + Platt hybrid calibrator (older behavior).',
    )
    return parser.parse_args()


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not path.exists():
        raise FileNotFoundError(f'Feature file not found: {path}')
    data = np.load(path, allow_pickle=True)
    paths = data['paths'] if 'paths' in data.files else None
    return data['features'], data['labels'], paths


def _subsample(
    features: np.ndarray,
    labels: np.ndarray,
    paths: np.ndarray | None,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(labels):
        return features, labels, paths
    idx = np.arange(len(labels))[:max_samples]
    sampled_paths = None if paths is None else paths[idx]
    return features[idx], labels[idx], sampled_paths


def _candidate_params(base_estimators: int, mode: str) -> list[dict]:
    fast_grid = [
        {
            'n_estimators': min(base_estimators, 400),
            'max_depth': 4,
            'learning_rate': 0.06,
            'subsample': 0.90,
            'colsample_bytree': 0.90,
            'min_child_weight': 2,
            'reg_lambda': 1.5,
        },
        {
            'n_estimators': min(base_estimators, 450),
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'reg_lambda': 2.0,
        },
    ]

    rigorous_grid = [
        {
            'n_estimators': base_estimators,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 2,
            'reg_lambda': 1.5,
        },
        {
            'n_estimators': base_estimators,
            'max_depth': 5,
            'learning_rate': 0.04,
            'subsample': 0.90,
            'colsample_bytree': 0.80,
            'min_child_weight': 3,
            'reg_lambda': 2.0,
        },
        {
            'n_estimators': base_estimators,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.90,
            'colsample_bytree': 0.90,
            'min_child_weight': 4,
            'reg_lambda': 2.5,
        },
    ]

    return fast_grid if mode == 'fast' else rigorous_grid


def build_model(params: dict, y_train: np.ndarray) -> XGBClassifier:
    y_train = np.asarray(y_train).astype(int)
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0

    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def _cnn_probs_for_paths(valid_paths: np.ndarray, y_valid: np.ndarray, cfg) -> np.ndarray:
    cnn_model = load_cnn_model(cfg.cnn_model_path)
    cnn_metadata = load_optional_json(cfg.cnn_metadata_path) or {}
    backbone = cnn_metadata.get('backbone_name', cnn_metadata.get('model_type', cfg.backbone_name))
    model_type = normalize_model_type(str(backbone))
    image_size = int(cnn_metadata.get('image_size', get_model_input_size(cnn_model)))
    dual = bool(cnn_metadata.get('dual_hybrid_backbone', False))
    class_rgb_stats = None
    stats_path = Path(cnn_metadata.get('class_rgb_stats_path', cfg.cnn_train_rgb_stats_path))
    if dual and stats_path.exists():
        try:
            class_rgb_stats = load_joblib(stats_path)
        except OSError:
            class_rgb_stats = None

    df = pd.DataFrame(
        {
            'abs_path': valid_paths.astype(str),
            'label': y_valid.astype(int),
            'label_str': np.where(y_valid.astype(int) == 1, 'real', 'fake'),
        }
    )

    seq = CSVImageSequence(
        dataframe=df,
        batch_size=cfg.batch_size,
        image_size=image_size,
        model_type=model_type,
        training=False,
        shuffle=False,
        seed=cfg.random_seed,
        aug_hflip_prob=cfg.aug_hflip_prob,
        aug_brightness_delta=cfg.aug_brightness_delta,
        aug_contrast_low=cfg.aug_contrast_low,
        aug_contrast_high=cfg.aug_contrast_high,
        cache_images=True,
        max_cache_images=cfg.max_cache_images,
        use_face_crop=cfg.use_face_crop,
        face_crop_expand=cfg.face_crop_expand,
        rgb_input_only=dual,
        auxiliary_quality=False,
        class_rgb_stats=class_rgb_stats,
    )

    raw_probs = predict_cnn_real_probs(cnn_model, seq, verbose=0)
    cnn_calibrator = load_optional_calibrator(cfg.cnn_calibrator_path)
    calibrated, _ = apply_calibration(raw_probs, cnn_calibrator)
    return calibrated


def _write_smart_router(
    path: Path,
    enabled: bool,
    strategy_payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump({'enabled': bool(enabled), **strategy_payload}, fp, indent=2)


def _train_svm(mode: str, x_train: np.ndarray, y_train: np.ndarray) -> SVC:
    if mode == 'fast':
        c_value = 3.0
    else:
        c_value = 6.0
    model = SVC(
        C=c_value,
        kernel='rbf',
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _fixed_xgb_for_hybrid(y_train: np.ndarray) -> XGBClassifier:
    y_train = np.asarray(y_train).astype(int)
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0
    return XGBClassifier(
        n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.85, colsample_bytree=0.75, reg_lambda=3.0, reg_alpha=0,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
    )


def _train_standard_hybrid(
    args: argparse.Namespace,
    cfg,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_paths: np.ndarray | None,
) -> None:
    mode = 'strong' if args.mode == 'rigorous' else args.mode

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    pca_path = Path(args.pca_path).resolve()
    xgb_base = _fixed_xgb_for_hybrid(y_train)
    fit_kwargs = {
        'X': x_train_scaled,
        'y': y_train,
        'eval_set': [(x_valid_scaled, y_valid)],
        'verbose': False,
    }
    try:
        xgb_base.fit(**fit_kwargs, early_stopping_rounds=args.early_stopping_rounds)
    except TypeError:
        xgb_base.fit(**fit_kwargs)

    raw_prob_valid = xgb_base.predict_proba(x_valid_scaled)[:, 1]
    hybrid_cal = fit_hybrid_calibrator_isotonic(y_true=y_valid, probs=raw_prob_valid)
    calibrated_prob_valid, _ = apply_calibration(raw_prob_valid, hybrid_cal)
    thr_bal, bal_stats = balanced_tpr_tnr_threshold(y_valid, calibrated_prob_valid)
    thr_opt, opt_stats = optimize_threshold(y_true=y_valid, y_prob=calibrated_prob_valid)
    hybrid_threshold = float(thr_opt)

    LOGGER.info(
        'Validation balanced TPR/TNR threshold=%.4f (tpr=%.3f tnr=%.3f); optimize_threshold=%.4f macro_f1=%.4f',
        hybrid_threshold,
        bal_stats.get('tpr', 0),
        bal_stats.get('tnr', 0),
        float(thr_opt),
        float(opt_stats.get('macro_f1', 0)),
    )

    calibrated_metrics = compute_classification_metrics(
        y_true=y_valid, y_prob=calibrated_prob_valid, threshold=hybrid_threshold
    )
    raw_metrics = compute_classification_metrics(y_true=y_valid, y_prob=raw_prob_valid)

    model_path = Path(args.output_model).resolve()
    scaler_path = Path(args.scaler_path).resolve()
    calibrator_path = Path(args.calibrator_path).resolve()
    metadata_path = Path(args.metadata_path).resolve()
    smart_router_path = Path(args.smart_router_path).resolve()

    save_xgb_model(xgb_base, model_path)
    save_joblib(scaler, scaler_path)
    save_calibrator(hybrid_cal, calibrator_path)

    metadata = {
        'mode': mode,
        'hybrid_pipeline': 'cnn_plus_identity_features_standard_scaler_no_pca_xgb_isotonic_valid',
        'final_classifier': 'xgboost',
        'calibration': 'isotonic_or_platt_on_validation',
        'use_platt_overlay': True,
        'pca_used': False,
        'pca_path': None,
        'pca_n_components': None,
        'pca_explained_variance_ratio_sum': None,
        'feature_scaled': True,
        'scaler_path': str(scaler_path),
        'calibrator_path': str(calibrator_path),
        'model_path': str(model_path),
        'decision_threshold': hybrid_threshold,
        'threshold_selection': {
            'primary': opt_stats,
            'balanced_tpr_tnr': bal_stats,
        },
        'xgboost_params': {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 2,
            'reg_alpha': 1,
        },
    }
    save_json_file(metadata, metadata_path)

    smart_payload = {
        'strategy': 'none',
        'reason': 'disabled_or_not_improved',
        'comparison': {},
    }

    if not args.disable_smart_router and valid_paths is not None and len(valid_paths) == len(y_valid):
        try:
            cnn_prob_valid = _cnn_probs_for_paths(valid_paths=valid_paths, y_valid=y_valid, cfg=cfg)

            cnn_meta = load_optional_json(cfg.cnn_metadata_path) or {}
            cnn_threshold = float(cnn_meta.get('decision_threshold', 0.5))
            cnn_metrics = compute_classification_metrics(y_true=y_valid, y_prob=cnn_prob_valid, threshold=cnn_threshold)

            fusion_fit = fit_class_aware_fusion(
                y_true=y_valid,
                cnn_prob=cnn_prob_valid,
                hybrid_prob=calibrated_prob_valid,
            )
            params = ClassAwareFusionParams(
                cnn_weight=float(fusion_fit['params']['cnn_weight']),
                real_gate=float(fusion_fit['params']['real_gate']),
                fake_gate=float(fusion_fit['params']['fake_gate']),
                decision_threshold=float(fusion_fit['params']['decision_threshold']),
            )
            smart_prob_valid = apply_class_aware_fusion(cnn_prob=cnn_prob_valid, hybrid_prob=calibrated_prob_valid, params=params)
            smart_metrics = compute_classification_metrics(
                y_true=y_valid,
                y_prob=smart_prob_valid,
                threshold=params.decision_threshold,
            )

            cnn_macro = float(cnn_metrics['per_class']['real']['f1_score'] + cnn_metrics['per_class']['fake']['f1_score']) / 2.0
            hybrid_macro = float(calibrated_metrics['per_class']['real']['f1_score'] + calibrated_metrics['per_class']['fake']['f1_score']) / 2.0
            smart_macro = float(smart_metrics['per_class']['real']['f1_score'] + smart_metrics['per_class']['fake']['f1_score']) / 2.0

            baseline_best = max(cnn_macro, hybrid_macro)
            improve_margin = 0.002
            enable_smart = smart_macro > (baseline_best + improve_margin)

            smart_payload = {
                'strategy': 'class_aware_fusion',
                'enabled': bool(enable_smart),
                'params': {
                    'cnn_weight': params.cnn_weight,
                    'real_gate': params.real_gate,
                    'fake_gate': params.fake_gate,
                    'decision_threshold': params.decision_threshold,
                },
                'comparison': {
                    'cnn_macro_f1': cnn_macro,
                    'hybrid_macro_f1': hybrid_macro,
                    'smart_macro_f1': smart_macro,
                    'improve_margin_required': improve_margin,
                },
                'metrics': {
                    'cnn_valid': cnn_metrics,
                    'hybrid_valid': calibrated_metrics,
                    'smart_valid': smart_metrics,
                },
            }
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('Smart router training skipped due to error: %s', exc)
            smart_payload = {
                'strategy': 'none',
                'reason': f'smart_router_error: {exc}',
                'comparison': {},
            }

    _write_smart_router(
        path=smart_router_path,
        enabled=bool(smart_payload.get('enabled', False)),
        strategy_payload=smart_payload,
    )

    metrics_path = Path(args.metrics_path).resolve()
    save_json(
        {
            'pipeline': 'standard_hybrid',
            'raw_like': raw_metrics,
            'calibrated': calibrated_metrics,
            'balanced_threshold': bal_stats,
            'optimize_threshold': opt_stats,
            'smart_router': smart_payload,
        },
        metrics_path,
    )

    LOGGER.info('Saved XGBoost base model to: %s', model_path)
    LOGGER.info('Saved validation sigmoid (Platt) calibrator to: %s', calibrator_path)
    LOGGER.info('Saved feature scaler to: %s', scaler_path)
    LOGGER.info('Saved PCA to: disabled')
    LOGGER.info('Hybrid validation metrics: %s', calibrated_metrics)
    LOGGER.info(
        'Hybrid validation FPR(real→fake)=%s FNR(fake→real)=%s CM=%s',
        calibrated_metrics.get('false_positive_rate_real'),
        calibrated_metrics.get('false_negative_rate_fake'),
        calibrated_metrics.get('confusion_matrix'),
    )


def _train_legacy_pipeline(
    args: argparse.Namespace,
    cfg,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    valid_paths: np.ndarray | None,
) -> None:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    mode = 'strong' if args.mode == 'rigorous' else args.mode
    best_model = None
    best_score = -1.0
    best_stats = None
    best_params = None
    best_name = None

    xgb_candidates = []
    for idx, params in enumerate(_candidate_params(args.n_estimators, 'fast' if mode == 'fast' else 'rigorous'), start=1):
        model = build_model(params, y_train=y_train)
        LOGGER.info('[%d] Training candidate params: %s', idx, params)
        fit_kwargs = {
            'X': x_train_scaled,
            'y': y_train,
            'eval_set': [(x_valid_scaled, y_valid)],
            'verbose': False,
        }
        try:
            model.fit(**fit_kwargs, early_stopping_rounds=args.early_stopping_rounds)
        except TypeError:
            model.fit(**fit_kwargs)
        prob_valid = model.predict_proba(x_valid_scaled)[:, 1]
        threshold, threshold_stats = optimize_threshold(y_true=y_valid, y_prob=prob_valid)

        LOGGER.info(
            '[%d] Validation macro-F1=%.5f accuracy=%.5f threshold=%.3f',
            idx,
            threshold_stats['macro_f1'],
            threshold_stats['accuracy'],
            threshold,
        )

        candidate_score = float(threshold_stats['macro_f1'])
        xgb_candidates.append((candidate_score, model, params, threshold_stats))

    if xgb_candidates:
        xgb_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_model, best_params, best_stats = xgb_candidates[0]
        best_name = 'xgboost'

    if not args.disable_svm:
        LOGGER.info('Training SVM candidate')
        svm_model = _train_svm(mode=mode, x_train=x_train_scaled, y_train=y_train)
        svm_prob_valid = svm_model.predict_proba(x_valid_scaled)[:, 1]
        svm_threshold, svm_stats = optimize_threshold(y_true=y_valid, y_prob=svm_prob_valid)
        LOGGER.info(
            '[svm] Validation macro-F1=%.5f accuracy=%.5f threshold=%.3f',
            svm_stats['macro_f1'],
            svm_stats['accuracy'],
            svm_threshold,
        )
        svm_score = float(svm_stats['macro_f1'])
        if svm_score > best_score:
            best_score = svm_score
            best_model = svm_model
            best_params = {'kernel': 'rbf', 'class_weight': 'balanced'}
            best_stats = svm_stats
            best_name = 'svm'

    if (not args.disable_ensemble) and (not args.disable_svm) and xgb_candidates:
        best_xgb_model = xgb_candidates[0][1]
        LOGGER.info('Training calibrated soft-voting ensemble candidate')
        ens_model = SoftVotingHybridEnsemble(models=[best_xgb_model, _train_svm(mode=mode, x_train=x_train_scaled, y_train=y_train)])
        ens_prob_valid = ens_model.predict_proba(x_valid_scaled)[:, 1]
        ens_threshold, ens_stats = optimize_threshold(y_true=y_valid, y_prob=ens_prob_valid)
        LOGGER.info(
            '[ensemble] Validation macro-F1=%.5f accuracy=%.5f threshold=%.3f',
            ens_stats['macro_f1'],
            ens_stats['accuracy'],
            ens_threshold,
        )
        ens_score = float(ens_stats['macro_f1'])
        if ens_score > best_score:
            best_score = ens_score
            best_model = ens_model
            best_params = {'type': 'soft_voting', 'members': ['xgboost', 'svm']}
            best_stats = ens_stats
            best_name = 'ensemble'

    if best_model is None or best_stats is None:
        raise RuntimeError('XGBoost training failed: no candidate model was selected.')

    raw_prob_valid = best_model.predict_proba(x_valid_scaled)[:, 1]
    calibrator = fit_hybrid_calibrator_isotonic(y_true=y_valid, probs=raw_prob_valid)
    calibrated_prob_valid, _ = apply_calibration(raw_prob_valid, calibrator)

    hybrid_threshold, hybrid_threshold_stats = optimize_threshold(y_true=y_valid, y_prob=calibrated_prob_valid)

    raw_metrics = compute_classification_metrics(y_true=y_valid, y_prob=raw_prob_valid)
    calibrated_metrics = compute_classification_metrics(y_true=y_valid, y_prob=calibrated_prob_valid, threshold=hybrid_threshold)

    model_path = Path(args.output_model).resolve()
    scaler_path = Path(args.scaler_path).resolve()
    calibrator_path = Path(args.calibrator_path).resolve()
    metadata_path = Path(args.metadata_path).resolve()
    smart_router_path = Path(args.smart_router_path).resolve()

    save_xgb_model(best_model, model_path)
    save_joblib(scaler, scaler_path)
    save_calibrator(calibrator, calibrator_path)

    metadata = {
        'mode': mode,
        'final_classifier': best_name,
        'best_params': best_params,
        'candidate_selection_metric': 'validation_macro_f1',
        'best_candidate_macro_f1': best_score,
        'feature_scaled': True,
        'use_platt_overlay': True,
        'scaler_path': str(scaler_path),
        'calibrator_path': str(calibrator_path),
        'model_path': str(model_path),
        'decision_threshold': float(hybrid_threshold),
        'threshold_selection': hybrid_threshold_stats,
    }
    save_json_file(metadata, metadata_path)

    smart_payload = {
        'strategy': 'none',
        'reason': 'disabled_or_not_improved',
        'comparison': {},
    }

    if not args.disable_smart_router and valid_paths is not None and len(valid_paths) == len(y_valid):
        try:
            cnn_prob_valid = _cnn_probs_for_paths(valid_paths=valid_paths, y_valid=y_valid, cfg=cfg)

            cnn_meta = load_optional_json(cfg.cnn_metadata_path) or {}
            cnn_threshold = float(cnn_meta.get('decision_threshold', 0.5))
            cnn_metrics = compute_classification_metrics(y_true=y_valid, y_prob=cnn_prob_valid, threshold=cnn_threshold)

            fusion_fit = fit_class_aware_fusion(
                y_true=y_valid,
                cnn_prob=cnn_prob_valid,
                hybrid_prob=calibrated_prob_valid,
            )
            params = ClassAwareFusionParams(
                cnn_weight=float(fusion_fit['params']['cnn_weight']),
                real_gate=float(fusion_fit['params']['real_gate']),
                fake_gate=float(fusion_fit['params']['fake_gate']),
                decision_threshold=float(fusion_fit['params']['decision_threshold']),
            )
            smart_prob_valid = apply_class_aware_fusion(cnn_prob=cnn_prob_valid, hybrid_prob=calibrated_prob_valid, params=params)
            smart_metrics = compute_classification_metrics(
                y_true=y_valid,
                y_prob=smart_prob_valid,
                threshold=params.decision_threshold,
            )

            cnn_macro = float(cnn_metrics['per_class']['real']['f1_score'] + cnn_metrics['per_class']['fake']['f1_score']) / 2.0
            hybrid_macro = float(calibrated_metrics['per_class']['real']['f1_score'] + calibrated_metrics['per_class']['fake']['f1_score']) / 2.0
            smart_macro = float(smart_metrics['per_class']['real']['f1_score'] + smart_metrics['per_class']['fake']['f1_score']) / 2.0

            baseline_best = max(cnn_macro, hybrid_macro)
            improve_margin = 0.002
            enable_smart = smart_macro > (baseline_best + improve_margin)

            smart_payload = {
                'strategy': 'class_aware_fusion',
                'enabled': bool(enable_smart),
                'params': {
                    'cnn_weight': params.cnn_weight,
                    'real_gate': params.real_gate,
                    'fake_gate': params.fake_gate,
                    'decision_threshold': params.decision_threshold,
                },
                'comparison': {
                    'cnn_macro_f1': cnn_macro,
                    'hybrid_macro_f1': hybrid_macro,
                    'smart_macro_f1': smart_macro,
                    'improve_margin_required': improve_margin,
                },
                'metrics': {
                    'cnn_valid': cnn_metrics,
                    'hybrid_valid': calibrated_metrics,
                    'smart_valid': smart_metrics,
                },
            }
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('Smart router training skipped due to error: %s', exc)
            smart_payload = {
                'strategy': 'none',
                'reason': f'smart_router_error: {exc}',
                'comparison': {},
            }

    _write_smart_router(
        path=smart_router_path,
        enabled=bool(smart_payload.get('enabled', False)),
        strategy_payload=smart_payload,
    )

    metrics_path = Path(args.metrics_path).resolve()
    save_json(
        {
            'raw': raw_metrics,
            'calibrated': calibrated_metrics,
            'selected_threshold': hybrid_threshold_stats,
            'smart_router': smart_payload,
        },
        metrics_path,
    )

    LOGGER.info('Saved XGBoost model to: %s', model_path)
    LOGGER.info('Saved feature scaler to: %s', scaler_path)
    LOGGER.info('Saved hybrid calibrator to: %s', calibrator_path)
    LOGGER.info('Saved smart router config to: %s', smart_router_path)
    LOGGER.info('Hybrid calibrated validation metrics: %s', calibrated_metrics)


def main() -> None:
    args = parse_args()
    cfg = get_config()
    cfg.ensure_output_dirs()

    feature_dir = Path(args.feature_dir).resolve()
    x_train, y_train, _ = load_npz(feature_dir / 'train_features.npz')
    x_valid, y_valid, valid_paths = load_npz(feature_dir / 'valid_features.npz')

    x_train, y_train, _ = _subsample(x_train, y_train, None, args.max_train_samples)
    x_valid, y_valid, valid_paths = _subsample(x_valid, y_valid, valid_paths, args.max_valid_samples)

    if args.legacy_ensemble:
        _train_legacy_pipeline(args, cfg, x_train, y_train, x_valid, y_valid, valid_paths)
    else:
        _train_standard_hybrid(args, cfg, x_train, y_train, x_valid, y_valid, valid_paths)


if __name__ == '__main__':
    main()
