from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

keras_home = PROJECT_ROOT / '.keras'
keras_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('KERAS_HOME', str(keras_home))
os.environ.setdefault('TFHUB_CACHE_DIR', str(keras_home / 'tfhub'))
if 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = os.environ.get('HDFD_KERAS_BACKEND', 'torch')

import argparse
import json
import logging

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from keras import mixed_precision

from utils.calibration_utils import apply_calibration, fit_hybrid_calibrator_isotonic, save_calibrator
from utils.config import AppConfig, get_config
from utils.data_loader import (
    SplitData,
    compute_class_rgb_mean_std,
    create_split_sequence,
    load_split_dataframe,
    validate_dataset,
)
from utils.fusion_utils import optimize_threshold
from utils.metrics_utils import compute_classification_metrics, save_json
from utils.model_utils import (
    BinaryFocalLoss,
    build_cnn_model,
    build_hybrid_dual_backbone_cnn,
    compile_cnn_model,
    compile_dual_hybrid_cnn_model,
    default_deep_tail_layers,
    default_image_size,
    default_tail_layers,
    is_dual_backbone_model,
    normalize_model_type,
    predict_cnn_real_probs,
    save_joblib,
    save_json_file,
    save_keras_model,
    set_seed,
    set_trainable_backbone_last_n,
    set_trainable_backbone_last_n_dual,
    set_trainable_for_finetune,
    set_trainable_for_full,
    set_trainable_for_full_dual,
    set_trainable_for_warmup,
    set_trainable_for_warmup_dual,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('train_cnn')


def _optional_bool(value: str | None):
    if value is None:
        return None
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Train baseline CNN deepfake detector from CSV splits.')

    parser.add_argument('--mode', choices=['debug', 'fast', 'strong', 'rigorous'], default='fast')
    parser.add_argument('--fast-mode', action='store_true')
    parser.add_argument('--rigorous-mode', action='store_true')

    parser.add_argument(
        '--backbone-name',
        type=str,
        default=None,
        choices=['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16'],
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default=None,
        choices=['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16'],
        help='Backward-compatible alias of --backbone-name.',
    )

    parser.add_argument('--epochs', type=int, default=cfg.epochs)
    parser.add_argument('--epochs-head', type=int, default=None)
    parser.add_argument('--epochs-finetune-stage2', type=int, default=None)
    parser.add_argument('--epochs-finetune-stage3', type=int, default=None)

    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=cfg.learning_rate)
    parser.add_argument('--head-learning-rate', type=float, default=None)
    parser.add_argument('--finetune-learning-rate', type=float, default=None)
    parser.add_argument('--deep-finetune-learning-rate', type=float, default=None)

    parser.add_argument('--dropout-rate', type=float, default=cfg.dropout_rate)
    parser.add_argument('--label-smoothing', type=float, default=cfg.label_smoothing)

    parser.add_argument('--tail-layers-stage2', type=int, default=None)
    parser.add_argument('--tail-layers-stage3', type=int, default=None)

    parser.add_argument('--use-mixed-precision', type=_optional_bool, default=None)
    parser.add_argument('--use-class-weights', type=_optional_bool, default=None)
    parser.add_argument('--cache-images', type=_optional_bool, default=None)
    parser.add_argument('--workers', type=int, default=None)

    parser.add_argument('--output-model', type=str, default=str(cfg.cnn_model_path))
    parser.add_argument('--history-json', type=str, default=str(cfg.history_json_path))
    parser.add_argument('--curve-path', type=str, default=str(cfg.training_curve_path))
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-valid-samples', type=int, default=None)
    parser.add_argument('--resume-if-exists', action='store_true', help='Resume from output-model checkpoint if it exists.')
    parser.add_argument('--resume-from', type=str, default=None, help='Explicit checkpoint path to resume from.')
    parser.add_argument(
        '--resume-stage',
        choices=['auto', 'head', 'stage2', 'stage3'],
        default='auto',
        help='When resuming: auto skips head and continues fine-tuning stages.',
    )
    parser.add_argument('--force-restart', action='store_true', help='Ignore checkpoints and restart training from scratch.')
    parser.add_argument(
        '--finetune-last-layers',
        type=int,
        default=30,
        help='EfficientNetB0 stage-2: unfreeze only the last N backbone layers (below global_avg_pool).',
    )
    parser.add_argument(
        '--no-dual-hybrid',
        action='store_true',
        help='Disable EfficientNetB0+MobileNetV2 dual backbone (single backbone only).',
    )
    parser.add_argument(
        '--skip-hard-negative-rounds',
        action='store_true',
        help='Do not append mined real false-positives for extra training passes.',
    )
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace, cfg: AppConfig) -> str:
    if args.rigorous_mode or cfg.rigorous_mode:
        return 'strong'
    if args.mode == 'rigorous':
        return 'strong'
    if args.fast_mode or cfg.fast_mode:
        return 'fast' if args.mode == 'fast' else args.mode
    return args.mode


def _mode_defaults(mode: str, backbone: str, cfg: AppConfig) -> dict:
    base_image_size = default_image_size(backbone)
    if 'efficientnet' in backbone:
        fast_image_size = 224
    else:
        fast_image_size = min(192, base_image_size)
    defaults = {
        'debug': {
            'batch_size': 16,
            'image_size': min(224, base_image_size),
            'epochs_head': 2,
            'epochs_stage2': 2,
            'epochs_stage3': 0,
            'max_train_samples': 1024,
            'max_valid_samples': 256,
            'patience': 3,
        },
        'fast': {
            'batch_size': 32,
            'image_size': fast_image_size,
            'epochs_head': 6,
            'epochs_stage2': 10,
            'epochs_stage3': 0,
            'max_train_samples': cfg.fast_max_train_samples,
            'max_valid_samples': cfg.fast_max_valid_samples,
            'patience': 3,
        },
        'strong': {
            'batch_size': 32,
            'image_size': base_image_size,
            'epochs_head': 5,
            'epochs_stage2': 15,
            'epochs_stage3': 0,
            'max_train_samples': None,
            'max_valid_samples': None,
            'patience': 3,
        },
    }
    return defaults[mode]


def _fit_with_worker_fallback(model, workers: int, **fit_kwargs):
    workers = max(1, int(workers))
    if workers <= 1:
        return model.fit(**fit_kwargs)

    try:
        return model.fit(
            workers=workers,
            use_multiprocessing=False,
            max_queue_size=max(8, workers * 2),
            **fit_kwargs,
        )
    except TypeError:
        LOGGER.warning('Current Keras backend does not accept workers in fit(); falling back to single-thread data loading.')
        return model.fit(**fit_kwargs)


def _configure_precision(use_mixed_precision: bool) -> str:
    has_gpu = bool(torch.cuda.is_available())
    if use_mixed_precision and has_gpu:
        policy = 'mixed_float16'
    else:
        policy = 'float32'
    mixed_precision.set_global_policy(policy)
    return policy


def save_training_curves(history_dict: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)

    axes[0].plot(history_dict.get('accuracy', []), label='train_accuracy', color='#63B3ED')
    axes[0].plot(history_dict.get('val_accuracy', []), label='val_accuracy', color='#F6AD55')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(history_dict.get('loss', []), label='train_loss', color='#68D391')
    axes[1].plot(history_dict.get('val_loss', []), label='val_loss', color='#FC8181')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def merge_histories(*histories) -> dict:
    merged: dict[str, list] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def class_weights_from_labels(labels: np.ndarray) -> dict[int, float]:
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    weights: dict[int, float] = {}
    for cls_idx, cls_count in enumerate(counts):
        if cls_count == 0:
            weights[cls_idx] = 1.0
        else:
            weights[cls_idx] = float(total / (len(counts) * cls_count))
    return weights


def _resolve_backbone(args: argparse.Namespace, cfg: AppConfig, mode: str) -> str:
    if args.backbone_name:
        return normalize_model_type(args.backbone_name)
    if args.model_type:
        return normalize_model_type(args.model_type)
    if mode == 'fast':
        return normalize_model_type(cfg.fast_backbone_name)
    if mode == 'strong':
        return normalize_model_type(cfg.strong_backbone_name)
    candidate = cfg.backbone_name or cfg.model_type
    return normalize_model_type(candidate)


def _infer_backbone_from_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    text = str(model_name).lower()
    for candidate in ['efficientnetb1', 'efficientnetb0', 'resnet50', 'mobilenetv2', 'vgg16']:
        if candidate in text:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    cfg = get_config()

    mode = _resolve_mode(args, cfg)
    backbone = _resolve_backbone(args, cfg, mode)
    mode_defaults = _mode_defaults(mode, backbone, cfg)
    use_dual_hybrid = bool(cfg.use_dual_hybrid_cnn) and not bool(args.no_dual_hybrid)

    set_seed(cfg.random_seed)
    cfg.ensure_output_dirs()

    batch_size = args.batch_size or mode_defaults['batch_size']
    image_size = args.image_size or mode_defaults['image_size']

    epochs_head = args.epochs_head if args.epochs_head is not None else mode_defaults['epochs_head']
    epochs_stage2 = args.epochs_finetune_stage2 if args.epochs_finetune_stage2 is not None else mode_defaults['epochs_stage2']
    epochs_stage3 = args.epochs_finetune_stage3 if args.epochs_finetune_stage3 is not None else mode_defaults['epochs_stage3']
    if backbone == 'efficientnetb0' and not use_dual_hybrid:
        epochs_stage3 = 0

    max_train_samples = args.max_train_samples if args.max_train_samples is not None else mode_defaults['max_train_samples']
    max_valid_samples = args.max_valid_samples if args.max_valid_samples is not None else mode_defaults['max_valid_samples']

    use_mixed_precision = cfg.use_mixed_precision if args.use_mixed_precision is None else args.use_mixed_precision
    use_class_weights = cfg.use_class_weights if args.use_class_weights is None else args.use_class_weights
    cache_images = cfg.cache_images if args.cache_images is None else args.cache_images
    workers = args.workers if args.workers is not None else cfg.dataloader_workers

    precision_policy = _configure_precision(use_mixed_precision=bool(use_mixed_precision))

    output_model_path = Path(args.output_model).resolve()
    resume_source = Path(args.resume_from).resolve() if args.resume_from else output_model_path
    resume_requested = (args.resume_if_exists or bool(args.resume_from)) and not args.force_restart
    resume_applied = False
    model: keras.Model | None = None

    if resume_requested and resume_source.exists():
        LOGGER.info('Resuming training from checkpoint: %s', resume_source)
        model = keras.models.load_model(
            str(resume_source),
            custom_objects={'BinaryFocalLoss': BinaryFocalLoss},
        )
        resume_applied = True
        use_dual_hybrid = is_dual_backbone_model(model)

        inferred_backbone = _infer_backbone_from_model_name(getattr(model, 'name', None))
        if inferred_backbone and inferred_backbone != backbone:
            LOGGER.warning(
                'Checkpoint backbone inferred as %s (requested %s). Using checkpoint backbone for resume.',
                inferred_backbone,
                backbone,
            )
            backbone = inferred_backbone

        model_input_shape = getattr(model, 'input_shape', None)
        if (
            isinstance(model_input_shape, tuple)
            and len(model_input_shape) >= 3
            and isinstance(model_input_shape[1], int)
            and model_input_shape[1] > 0
        ):
            checkpoint_image_size = int(model_input_shape[1])
            if checkpoint_image_size != image_size:
                LOGGER.warning(
                    'Checkpoint expects image_size=%d (requested %d). Using checkpoint size for resume.',
                    checkpoint_image_size,
                    image_size,
                )
                image_size = checkpoint_image_size

    if max_train_samples is None and max_valid_samples is None:
        validation_summary = validate_dataset(cfg, strict_missing_ratio=cfg.max_missing_ratio)
        LOGGER.info('Dataset validation summary: %s', validation_summary)
    else:
        LOGGER.info(
            'Skipping full dataset validation for sampled run '
            '(max_train_samples=%s, max_valid_samples=%s).',
            max_train_samples,
            max_valid_samples,
        )

    train_split = load_split_dataframe('train', cfg, max_samples=max_train_samples)
    valid_split = load_split_dataframe('valid', cfg, max_samples=max_valid_samples)

    class_rgb_stats = None
    if cfg.match_fake_to_real_stats and use_dual_hybrid:
        LOGGER.info('Computing per-class RGB mean/std for distribution matching (train split)...')
        class_rgb_stats = compute_class_rgb_mean_std(
            train_split.dataframe,
            image_size=image_size,
            use_face_crop=cfg.use_face_crop,
            face_crop_expand=cfg.face_crop_expand,
        )
        LOGGER.info('Class RGB stats keys=%s', list(class_rgb_stats.keys()))
        save_joblib(class_rgb_stats, cfg.cnn_train_rgb_stats_path)

    train_seq = create_split_sequence(
        split_data=train_split,
        config=cfg,
        model_type=backbone,
        training=True,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=False,
        rgb_input_only=use_dual_hybrid,
        auxiliary_quality=use_dual_hybrid,
        class_rgb_stats=class_rgb_stats,
    )
    valid_seq = create_split_sequence(
        split_data=valid_split,
        config=cfg,
        model_type=backbone,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=cache_images,
        rgb_input_only=use_dual_hybrid,
        auxiliary_quality=use_dual_hybrid,
        class_rgb_stats=class_rgb_stats,
    )

    LOGGER.info(
        'Training mode=%s backbone=%s dual_hybrid=%s policy=%s',
        mode,
        backbone,
        use_dual_hybrid,
        precision_policy,
    )
    LOGGER.info('Resolved image root: %s', train_split.image_root)
    LOGGER.info('Train rows=%d | Valid rows=%d', train_split.available_rows, valid_split.available_rows)

    class_weights = class_weights_from_labels(train_seq.get_labels()) if use_class_weights else None
    LOGGER.info('Class weights enabled=%s values=%s', bool(use_class_weights), class_weights)
    fit_class_weight = None
    if class_weights is not None:
        fit_class_weight = {'prediction': class_weights} if use_dual_hybrid else class_weights

    if model is None:
        if resume_requested:
            LOGGER.warning('Resume requested but checkpoint not found: %s. Starting fresh training.', resume_source)
        if use_dual_hybrid:
            model = build_hybrid_dual_backbone_cnn(
                image_size=image_size,
                learning_rate=args.learning_rate,
                dropout_rate=args.dropout_rate,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                label_smoothing=args.label_smoothing,
                aux_loss_weight=cfg.aux_loss_weight,
            )
        else:
            model = build_cnn_model(
                model_type=backbone,
                image_size=image_size,
                learning_rate=args.learning_rate,
                dropout_rate=args.dropout_rate,
                label_smoothing=args.label_smoothing,
            )

    val_auc_metric = 'val_prediction_auc' if use_dual_hybrid else 'val_auc'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_model_path),
            monitor=val_auc_metric,
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=val_auc_metric,
            patience=min(3, mode_defaults['patience']),
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(2, mode_defaults['patience'] // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    pretrained_loaded = bool(getattr(model, '_hdfd_pretrained_backbone', False))
    if resume_applied and not pretrained_loaded:
        pretrained_loaded = True
    if not pretrained_loaded:
        LOGGER.warning('Backbone is randomly initialized; rigorous transfer-learning behavior will be limited.')

    head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    stage2_lr = args.finetune_learning_rate if args.finetune_learning_rate is not None else max(1e-6, args.learning_rate * 0.5)
    stage3_lr = (
        args.deep_finetune_learning_rate
        if args.deep_finetune_learning_rate is not None
        else max(1e-6, args.learning_rate * 0.25)
    )

    tail_stage2 = args.tail_layers_stage2 if args.tail_layers_stage2 is not None else default_tail_layers(backbone)
    tail_stage3 = args.tail_layers_stage3 if args.tail_layers_stage3 is not None else default_deep_tail_layers(backbone)

    run_head = epochs_head > 0
    run_stage2 = epochs_stage2 > 0
    run_stage3 = epochs_stage3 > 0

    if resume_applied:
        if args.resume_stage == 'auto':
            run_head = False
            if not run_stage2 and not run_stage3 and epochs_head > 0:
                run_head = True
        elif args.resume_stage == 'head':
            run_head = epochs_head > 0
            run_stage2 = epochs_stage2 > 0
            run_stage3 = epochs_stage3 > 0
        elif args.resume_stage == 'stage2':
            run_head = False
            run_stage2 = epochs_stage2 > 0
            run_stage3 = epochs_stage3 > 0
        else:
            run_head = False
            run_stage2 = False
            run_stage3 = epochs_stage3 > 0

    LOGGER.info(
        'Stage plan (resume=%s, resume_stage=%s): head=%s stage2=%s stage3=%s',
        resume_applied,
        args.resume_stage,
        run_head,
        run_stage2,
        run_stage3,
    )

    histories: list = []

    if run_head:
        if use_dual_hybrid:
            if pretrained_loaded:
                set_trainable_for_warmup_dual(model)
            else:
                set_trainable_for_full_dual(model)
            compile_dual_hybrid_cnn_model(
                model=model,
                learning_rate=head_lr,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                label_smoothing=args.label_smoothing,
                aux_loss_weight=cfg.aux_loss_weight,
            )
        else:
            if pretrained_loaded:
                set_trainable_for_warmup(model)
            else:
                set_trainable_for_full(model)
            compile_cnn_model(
                model=model,
                learning_rate=head_lr,
                label_smoothing=args.label_smoothing,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
            )
        LOGGER.info('Stage 1/3 head training epochs=%d lr=%.6f', epochs_head, head_lr)
        histories.append(
            _fit_with_worker_fallback(
                model,
                workers=workers,
                x=train_seq,
                validation_data=valid_seq,
                epochs=epochs_head,
                callbacks=callbacks,
                class_weight=fit_class_weight,
                verbose=1,
            )
        )

    if run_stage2:
        if use_dual_hybrid:
            if pretrained_loaded:
                set_trainable_backbone_last_n_dual(
                    model,
                    n_eff=int(args.finetune_last_layers),
                    n_mob=max(30, int(args.finetune_last_layers)),
                )
            else:
                set_trainable_for_full_dual(model)
            compile_dual_hybrid_cnn_model(
                model=model,
                learning_rate=stage2_lr,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                label_smoothing=args.label_smoothing,
                aux_loss_weight=cfg.aux_loss_weight,
            )
        else:
            if pretrained_loaded and backbone == 'efficientnetb0':
                set_trainable_backbone_last_n(model, args.finetune_last_layers)
            elif pretrained_loaded:
                set_trainable_for_finetune(model, tail_layers=tail_stage2)
            else:
                set_trainable_for_full(model)
            compile_cnn_model(
                model=model,
                learning_rate=stage2_lr,
                label_smoothing=args.label_smoothing,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
            )
        LOGGER.info(
            'Stage 2/3 finetune epochs=%d backbone=%s tail_or_lastN=%s lr=%.6f',
            epochs_stage2,
            backbone,
            args.finetune_last_layers if backbone == 'efficientnetb0' else tail_stage2,
            stage2_lr,
        )
        histories.append(
            _fit_with_worker_fallback(
                model,
                workers=workers,
                x=train_seq,
                validation_data=valid_seq,
                epochs=epochs_stage2,
                callbacks=callbacks,
                class_weight=fit_class_weight,
                verbose=1,
            )
        )

    if run_stage3:
        if use_dual_hybrid:
            if pretrained_loaded:
                set_trainable_backbone_last_n_dual(
                    model,
                    n_eff=min(int(tail_stage3) * 2, 120),
                    n_mob=min(int(tail_stage3) * 2, 120),
                )
            else:
                set_trainable_for_full_dual(model)
            compile_dual_hybrid_cnn_model(
                model=model,
                learning_rate=stage3_lr,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
                label_smoothing=args.label_smoothing,
                aux_loss_weight=cfg.aux_loss_weight,
            )
        else:
            if pretrained_loaded:
                set_trainable_for_finetune(model, tail_layers=tail_stage3)
            else:
                set_trainable_for_full(model)
            compile_cnn_model(
                model=model,
                learning_rate=stage3_lr,
                label_smoothing=args.label_smoothing,
                focal_alpha=cfg.focal_alpha,
                focal_gamma=cfg.focal_gamma,
            )
        LOGGER.info('Stage 3/3 deeper finetune epochs=%d tail_layers=%d lr=%.6f', epochs_stage3, tail_stage3, stage3_lr)
        histories.append(
            _fit_with_worker_fallback(
                model,
                workers=workers,
                x=train_seq,
                validation_data=valid_seq,
                epochs=epochs_stage3,
                callbacks=callbacks,
                class_weight=fit_class_weight,
                verbose=1,
            )
        )

    # Automated hard-negative mining + extra training when real→fake FPR is high (reduces false positives).
    anti_fp_round = 0
    while (
        not args.skip_hard_negative_rounds
        and use_dual_hybrid
        and anti_fp_round < int(cfg.max_hard_negative_rounds)
    ):
        valid_probs_tmp = predict_cnn_real_probs(model, valid_seq, verbose=0)
        valid_labels_tmp = valid_seq.get_labels()
        vm_tmp = compute_classification_metrics(y_true=valid_labels_tmp, y_prob=valid_probs_tmp, threshold=0.5)
        fpr = vm_tmp.get('false_positive_rate_real')
        LOGGER.info(
            '[Anti-FP] round=%d validation FPR(real→fake)=%s AUC=%s',
            anti_fp_round,
            fpr,
            vm_tmp.get('roc_auc'),
        )
        if fpr is None or float(fpr) <= float(cfg.hybrid_fpr_retrain_threshold):
            break
        vm = (valid_labels_tmp == 1) & (valid_probs_tmp < 0.5)
        hard = valid_split.dataframe.iloc[np.where(vm)[0]].copy()
        if hard.empty:
            LOGGER.info('[Anti-FP] No misclassified REAL rows on validation; stopping.')
            break
        LOGGER.info('[Anti-FP] Appending %d hard REAL samples to training set.', len(hard))
        merged_df = pd.concat([train_split.dataframe, hard], ignore_index=True)
        real_only = train_split.dataframe.loc[train_split.dataframe['label'].astype(int) == 1]
        if len(real_only) > 0:
            n_dup = min(max(len(hard) * 3, 64), 6000, len(real_only) * 5)
            dup = real_only.sample(n=int(n_dup), replace=True, random_state=cfg.random_seed + anti_fp_round)
            merged_df = pd.concat([merged_df, dup], ignore_index=True)
        train_split = SplitData(
            split='train',
            dataframe=merged_df.reset_index(drop=True),
            image_root=train_split.image_root,
            total_rows=len(merged_df),
            available_rows=len(merged_df),
            missing_rows=0,
            label_distribution=merged_df['label_str'].value_counts().to_dict(),
            label_folder_mismatches=train_split.label_folder_mismatches,
        )
        train_seq = create_split_sequence(
            split_data=train_split,
            config=cfg,
            model_type=backbone,
            training=True,
            shuffle=True,
            batch_size=batch_size,
            image_size=image_size,
            cache_images=False,
            rgb_input_only=use_dual_hybrid,
            auxiliary_quality=use_dual_hybrid,
            class_rgb_stats=class_rgb_stats,
        )
        fit_class_weight_round = None
        if class_weights is not None:
            fit_class_weight_round = {'prediction': class_weights_from_labels(train_seq.get_labels())}
        extra_epochs = max(4, min(epochs_stage2 if epochs_stage2 > 0 else 8, 14))
        compile_dual_hybrid_cnn_model(
            model=model,
            learning_rate=max(1e-6, stage2_lr * 0.5),
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            label_smoothing=args.label_smoothing,
            aux_loss_weight=cfg.aux_loss_weight,
        )
        histories.append(
            _fit_with_worker_fallback(
                model,
                workers=workers,
                x=train_seq,
                validation_data=valid_seq,
                epochs=extra_epochs,
                callbacks=callbacks,
                class_weight=fit_class_weight_round,
                verbose=1,
            )
        )
        anti_fp_round += 1

    model = keras.models.load_model(str(output_model_path), custom_objects={'BinaryFocalLoss': BinaryFocalLoss})
    save_keras_model(model, output_model_path)

    history_dict = merge_histories(*histories)

    history_json_path = Path(args.history_json).resolve()
    history_json_path.parent.mkdir(parents=True, exist_ok=True)
    with history_json_path.open('w', encoding='utf-8') as fp:
        json.dump(history_dict, fp, indent=2)

    save_training_curves(history_dict, Path(args.curve_path).resolve())

    valid_probs_raw = predict_cnn_real_probs(model, valid_seq, verbose=0)
    valid_labels = valid_seq.get_labels()
    valid_metrics_raw = compute_classification_metrics(y_true=valid_labels, y_prob=valid_probs_raw)
    valid_pred_std = float(np.std(valid_probs_raw))
    valid_pred_auc = float(valid_metrics_raw['roc_auc'] or 0.0)
    if valid_pred_std < 1e-4 or valid_pred_auc < 0.52:
        raise RuntimeError(
            f'CNN collapse detected: validation prediction std={valid_pred_std:.8f}, roc_auc={valid_pred_auc:.4f}. '
            'Increase fine-tuning strength or switch backbone before feature extraction.'
        )

    calibrator = fit_hybrid_calibrator_isotonic(y_true=valid_labels, probs=valid_probs_raw)
    save_calibrator(calibrator, cfg.cnn_calibrator_path)

    valid_probs_cal, _ = apply_calibration(valid_probs_raw, calibrator)

    cnn_threshold, threshold_stats = optimize_threshold(y_true=valid_labels, y_prob=valid_probs_cal)
    valid_metrics_cal = compute_classification_metrics(y_true=valid_labels, y_prob=valid_probs_cal, threshold=cnn_threshold)
    valid_pred_cal = (valid_probs_cal >= cnn_threshold).astype(int)
    valid_pred_real_ratio = float(np.mean(valid_pred_cal == 1))
    valid_pred_fake_ratio = float(np.mean(valid_pred_cal == 0))
    valid_majority_ratio = float(max(valid_pred_real_ratio, valid_pred_fake_ratio))
    if valid_majority_ratio > 0.85:
        raise RuntimeError(
            f'Collapse guard triggered: model predicts one class for {valid_majority_ratio:.2%} '
            'of mixed validation images (>85%). Rejecting model.'
        )

    train_eval_seq = create_split_sequence(
        split_data=train_split,
        config=cfg,
        model_type=backbone,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=True,
        rgb_input_only=use_dual_hybrid,
        auxiliary_quality=False,
        class_rgb_stats=class_rgb_stats,
    )
    train_probs_raw = predict_cnn_real_probs(model, train_eval_seq, verbose=0)
    train_labels = train_eval_seq.get_labels()
    train_probs_cal, _ = apply_calibration(train_probs_raw, calibrator)
    train_metrics_raw = compute_classification_metrics(y_true=train_labels, y_prob=train_probs_raw)
    train_metrics_cal = compute_classification_metrics(y_true=train_labels, y_prob=train_probs_cal, threshold=cnn_threshold)

    save_json(
        {
            'train_raw': train_metrics_raw,
            'train_calibrated': train_metrics_cal,
            'valid_raw': valid_metrics_raw,
            'valid_calibrated': valid_metrics_cal,
            'selected_threshold': threshold_stats,
            'false_positive_rate_real_raw': valid_metrics_raw.get('false_positive_rate_real'),
            'false_positive_rate_real_calibrated': valid_metrics_cal.get('false_positive_rate_real'),
            'confusion_matrix_valid_calibrated': valid_metrics_cal.get('confusion_matrix'),
        },
        cfg.results_dir / 'cnn_valid_metrics.json',
    )

    metadata = {
        'model_type': 'hybrid_dual_efficientnet_mobilenet' if use_dual_hybrid else backbone,
        'backbone_name': backbone,
        'dual_hybrid_backbone': bool(use_dual_hybrid),
        'image_size': int(image_size),
        'dropout_rate': float(args.dropout_rate),
        'label_smoothing': float(args.label_smoothing),
        'mode': mode,
        'precision_policy': precision_policy,
        'epochs_head': int(epochs_head),
        'epochs_finetune_stage2': int(epochs_stage2),
        'epochs_finetune_stage3': int(epochs_stage3),
        'tail_layers_stage2': int(tail_stage2),
        'tail_layers_stage3': int(tail_stage3),
        'class_weights': class_weights,
        'pretrained_backbone_loaded': pretrained_loaded,
        'cnn_model_path': str(output_model_path),
        'cnn_calibrator_path': str(cfg.cnn_calibrator_path),
        'decision_threshold': float(cnn_threshold),
        'threshold_selection': threshold_stats,
        'collapse_guard': {
            'max_one_class_ratio': 0.85,
            'validation_pred_real_ratio': valid_pred_real_ratio,
            'validation_pred_fake_ratio': valid_pred_fake_ratio,
            'validation_majority_ratio': valid_majority_ratio,
            'passed': bool(valid_majority_ratio <= 0.85),
        },
        'focal_alpha': float(cfg.focal_alpha),
        'focal_gamma': float(cfg.focal_gamma),
        'aux_loss_weight': float(cfg.aux_loss_weight),
        'cnn_calibration': 'isotonic_or_platt_auto',
    }
    if class_rgb_stats is not None:
        metadata['class_rgb_stats_path'] = str(cfg.cnn_train_rgb_stats_path.resolve())
    save_json_file(metadata, cfg.cnn_metadata_path)

    LOGGER.info('Saved baseline CNN model to: %s', output_model_path)
    LOGGER.info('Saved CNN calibrator to: %s', cfg.cnn_calibrator_path)
    LOGGER.info('Train metrics (raw): %s', train_metrics_raw)
    LOGGER.info('Validation metrics (raw): %s', valid_metrics_raw)
    LOGGER.info('Validation metrics (calibrated): %s', valid_metrics_cal)
    LOGGER.info(
        'Validation FPR(real→fake) raw=%s calibrated=%s | AUC raw=%s',
        valid_metrics_raw.get('false_positive_rate_real'),
        valid_metrics_cal.get('false_positive_rate_real'),
        valid_metrics_raw.get('roc_auc'),
    )


if __name__ == '__main__':
    main()

