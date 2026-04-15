from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import subprocess

from utils.config import get_config

BACKBONE_CHOICES = ['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(
        description='Run full training pipeline: sanity -> (optional overfit) -> CNN -> features -> XGBoost'
    )
    parser.add_argument('--mode', choices=['debug', 'fast', 'strong', 'rigorous'], default='fast')
    parser.add_argument('--backbone-name', choices=BACKBONE_CHOICES, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=cfg.learning_rate)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-valid-samples', type=int, default=None)
    parser.add_argument('--max-test-samples', type=int, default=None)
    parser.add_argument('--run-overfit-check', action='store_true')
    parser.add_argument('--check-corrupt', action='store_true')
    parser.add_argument('--max-corrupt-checks-per-split', type=int, default=None)
    parser.add_argument('--force-feature-refresh', action='store_true')
    parser.add_argument('--disable-smart-router', action='store_true')
    parser.add_argument('--resume-from-checkpoint', action='store_true', help='Resume CNN checkpoint instead of strict fresh retrain.')
    parser.add_argument('--skip-evaluate', action='store_true', help='Do not run evaluate.py after the hybrid stage.')
    parser.add_argument(
        '--no-dual-hybrid',
        action='store_true',
        help='Train single-backbone CNN instead of EfficientNetB0+MobileNetV2 dual hybrid.',
    )
    parser.add_argument(
        '--skip-automated-validation',
        action='store_true',
        help='Skip Wikipedia/TPDNE automated_validation.py (requires network).',
    )
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    printable = ' '.join(command)
    print(f'\n[RUN] {printable}\n')

    env = os.environ.copy()
    keras_home = PROJECT_ROOT / '.keras'
    keras_home.mkdir(parents=True, exist_ok=True)
    env.setdefault('KERAS_HOME', str(keras_home))
    env.setdefault('TFHUB_CACHE_DIR', str(keras_home / 'tfhub'))

    subprocess.run(command, check=True, env=env)


def _optional_arg(flag: str, value) -> list[str]:
    if value is None:
        return []
    return [flag, str(value)]


def _resolve_backbone(mode: str, cfg, explicit: str | None) -> str:
    if explicit:
        return explicit
    if mode == 'fast':
        return cfg.fast_backbone_name
    if mode in {'strong', 'rigorous'}:
        return cfg.strong_backbone_name
    return cfg.backbone_name


def main() -> None:
    args = parse_args()
    cfg = get_config()

    py = sys.executable
    project_root = Path(__file__).resolve().parent
    backbone = _resolve_backbone(args.mode, cfg, args.backbone_name)

    if args.mode == 'fast' and args.max_train_samples is None:
        args.max_train_samples = cfg.fast_max_train_samples
    if args.mode == 'fast' and args.max_valid_samples is None:
        args.max_valid_samples = cfg.fast_max_valid_samples

    sanity_cmd = [
        py,
        str(project_root / 'verify_dataset.py'),
        '--output-json',
        str(cfg.dataset_audit_path),
    ]
    if args.check_corrupt:
        sanity_cmd.append('--check-corrupt')
    sanity_cmd += _optional_arg('--max-corrupt-checks-per-split', args.max_corrupt_checks_per_split)

    overfit_cmd = [
        py,
        str(project_root / 'tiny_overfit.py'),
        '--backbone-name',
        backbone,
    ]

    cnn_cmd = [
        py,
        str(project_root / 'train_cnn.py'),
        '--mode',
        args.mode,
        '--backbone-name',
        backbone,
        '--learning-rate',
        str(args.learning_rate),
        '--output-model',
        str(cfg.cnn_model_path),
    ]
    if not args.resume_from_checkpoint:
        cnn_cmd.append('--force-restart')
    cnn_cmd += _optional_arg('--batch-size', args.batch_size)
    cnn_cmd += _optional_arg('--image-size', args.image_size)
    cnn_cmd += _optional_arg('--max-train-samples', args.max_train_samples)
    cnn_cmd += _optional_arg('--max-valid-samples', args.max_valid_samples)
    if args.no_dual_hybrid:
        cnn_cmd.append('--no-dual-hybrid')

    feature_cmd = [
        py,
        str(project_root / 'extract_features.py'),
        '--model-path',
        str(cfg.cnn_model_path),
        '--feature-dir',
        str(cfg.feature_dir),
        '--model-type',
        backbone,
        '--reuse-existing',
    ]
    if args.force_feature_refresh:
        feature_cmd.append('--force')
    feature_cmd += _optional_arg('--batch-size', args.batch_size)
    feature_cmd += _optional_arg('--max-train-samples', args.max_train_samples)
    feature_cmd += _optional_arg('--max-valid-samples', args.max_valid_samples)
    feature_cmd += _optional_arg('--max-test-samples', args.max_test_samples)

    xgb_mode = 'fast' if args.mode in {'debug', 'fast'} else 'strong'
    xgb_cmd = [
        py,
        str(project_root / 'train_xgboost.py'),
        '--mode',
        xgb_mode,
        '--feature-dir',
        str(cfg.feature_dir),
        '--output-model',
        str(cfg.xgb_model_path),
        '--scaler-path',
        str(cfg.xgb_scaler_path),
        '--pca-path',
        str(cfg.hybrid_pca_path),
        '--calibrator-path',
        str(cfg.hybrid_calibrator_path),
        '--metadata-path',
        str(cfg.hybrid_metadata_path),
        '--smart-router-path',
        str(cfg.smart_router_path),
    ]
    if args.disable_smart_router:
        xgb_cmd.append('--disable-smart-router')

    run_step(sanity_cmd)
    if args.run_overfit_check:
        run_step(overfit_cmd)
    run_step(cnn_cmd)
    run_step(feature_cmd)
    run_step(xgb_cmd)

    if not args.skip_evaluate:
        eval_cmd = [py, str(project_root / 'evaluate.py')]
        run_step(eval_cmd)
        viz_cmd = [py, str(project_root / 'visualize_embeddings.py'), '--split', 'valid']
        run_step(viz_cmd)

    if not args.skip_automated_validation:
        autoval_cmd = [py, str(project_root / 'automated_validation.py')]
        print('\n[RUN] automated_validation (Wikipedia / TPDNE)\n')
        result = subprocess.run(autoval_cmd, env=os.environ.copy(), check=False)
        if result.returncode != 0:
            print(
                f'Warning: automated_validation exited with code {result.returncode} '
                '(often network or rate limits). See script output above.'
            )

    print('\nPipeline completed.')
    print(f'CNN model: {cfg.cnn_model_path}')
    print(f'CNN calibrator: {cfg.cnn_calibrator_path}')
    print(f'XGBoost model: {cfg.xgb_model_path}')
    print(f'XGBoost scaler: {cfg.xgb_scaler_path}')
    print(f'Hybrid calibrator: {cfg.hybrid_calibrator_path}')
    print(f'Smart router: {cfg.smart_router_path}')


if __name__ == '__main__':
    main()
