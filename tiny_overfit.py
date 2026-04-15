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
from keras import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC, BinaryAccuracy, Precision, Recall
from keras.optimizers import Adam

from utils.config import get_config
from utils.data_loader import SplitData, create_split_sequence, load_split_dataframe
from utils.metrics_utils import compute_classification_metrics
from utils.model_utils import build_cnn_model, default_image_size, normalize_model_type, set_seed

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('tiny_overfit')

BACKBONE_CHOICES = ['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Tiny-subset overfit sanity check for the CNN training pipeline.')
    parser.add_argument('--model-kind', choices=['simple', 'backbone'], default='simple')
    parser.add_argument('--backbone-name', type=str, default='mobilenetv2', choices=BACKBONE_CHOICES)
    parser.add_argument('--subset-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--target-fit-accuracy', type=float, default=0.92)
    parser.add_argument('--target-infer-accuracy', type=float, default=0.90)
    parser.add_argument('--report-path', type=str, default=str(cfg.results_dir / 'tiny_overfit_report.json'))
    return parser.parse_args()


def _freeze_batchnorm_layers(model: keras.Model) -> None:
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def _build_simple_overfit_model(image_size: int, learning_rate: float) -> keras.Model:
    model = Sequential(
        [
            Input(shape=(image_size, image_size, 3)),
            Conv2D(16, 3, activation='relu', padding='same'),
            MaxPooling2D(),
            Conv2D(32, 3, activation='relu', padding='same'),
            MaxPooling2D(),
            Conv2D(64, 3, activation='relu', padding='same'),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid'),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(),
        metrics=[
            BinaryAccuracy(name='accuracy'),
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
        ],
    )
    return model


def main() -> None:
    args = parse_args()
    cfg = get_config()

    set_seed(cfg.random_seed)

    backbone = normalize_model_type(args.backbone_name)
    image_size = args.image_size or min(192, default_image_size(backbone))

    train_split = load_split_dataframe('train', cfg, max_samples=args.subset_size)

    overfit_split = SplitData(
        split='tiny_overfit',
        dataframe=train_split.dataframe.copy(),
        image_root=train_split.image_root,
        total_rows=train_split.available_rows,
        available_rows=train_split.available_rows,
        missing_rows=0,
        label_distribution=train_split.label_distribution,
        label_folder_mismatches=train_split.label_folder_mismatches,
    )

    train_seq = create_split_sequence(
        split_data=overfit_split,
        config=cfg,
        model_type=backbone,
        training=False,
        shuffle=True,
        batch_size=args.batch_size,
        image_size=image_size,
        cache_images=True,
    )
    eval_seq = create_split_sequence(
        split_data=overfit_split,
        config=cfg,
        model_type=backbone,
        training=False,
        shuffle=False,
        batch_size=args.batch_size,
        image_size=image_size,
        cache_images=True,
    )

    if args.model_kind == 'simple':
        model = _build_simple_overfit_model(image_size=image_size, learning_rate=args.learning_rate)
    else:
        effective_lr = min(args.learning_rate, 1e-4)
        model = build_cnn_model(
            model_type=backbone,
            image_size=image_size,
            learning_rate=effective_lr,
            dropout_rate=0.1,
            label_smoothing=0.0,
        )
        # Tiny-subset sanity check is more stable with a staged transfer-learning schedule.
        for layer in model.layers:
            layer.trainable = True
        for layer in model.layers[:-3]:
            layer.trainable = False
        _freeze_batchnorm_layers(model)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='accuracy',
            patience=8,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        train_seq,
        validation_data=eval_seq,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    infer_probs = model.predict(eval_seq, verbose=0).ravel()
    infer_labels = eval_seq.get_labels()
    infer_metrics = compute_classification_metrics(y_true=infer_labels, y_prob=infer_probs)

    best_fit_acc = float(max(history.history.get('accuracy', [0.0])))
    infer_acc = float(infer_metrics['accuracy'])

    report = {
        'model_kind': args.model_kind,
        'backbone_name': backbone,
        'subset_size': int(args.subset_size),
        'epochs_requested': int(args.epochs),
        'image_size': int(image_size),
        'best_fit_accuracy': best_fit_acc,
        'inference_accuracy': infer_acc,
        'target_fit_accuracy': float(args.target_fit_accuracy),
        'target_infer_accuracy': float(args.target_infer_accuracy),
        'metrics': infer_metrics,
        'passed': bool(best_fit_acc >= args.target_fit_accuracy and infer_acc >= args.target_infer_accuracy),
    }

    report_path = Path(args.report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2)

    LOGGER.info('Tiny overfit report saved to: %s', report_path)
    LOGGER.info('Tiny overfit summary: %s', report)

    if not report['passed']:
        raise SystemExit(
            'Overfit check failed: expected high fit and inference accuracy on tiny subset. '
            f"got fit={best_fit_acc:.4f}, infer={infer_acc:.4f}."
        )


if __name__ == '__main__':
    main()
