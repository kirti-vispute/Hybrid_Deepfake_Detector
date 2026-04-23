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

from utils.classical_features import IdentityFeatureSpec, extract_identity_features_for_paths
from utils.config import get_config
from utils.data_loader import create_split_sequence, load_split_dataframe
from utils.model_utils import (
    get_feature_extractor,
    get_model_input_size,
    load_cnn_model,
    load_joblib,
    load_optional_json,
    normalize_model_type,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('extract_features')


BACKBONE_CHOICES = ['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']
IDENTITY_SPEC = IdentityFeatureSpec()


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Extract CNN penultimate-layer features for train/valid/test splits.')
    parser.add_argument('--model-path', type=str, default=str(cfg.cnn_model_path))
    parser.add_argument('--feature-dir', type=str, default=str(cfg.feature_dir))
    parser.add_argument('--extractor-config', type=str, default=str(cfg.extractor_config_path))
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--model-type', type=str, default=cfg.backbone_name, choices=BACKBONE_CHOICES)
    parser.add_argument('--cache-images', action='store_true')
    parser.add_argument('--reuse-existing', action='store_true', help='Reuse existing .npz files when row counts match expected split size.')
    parser.add_argument('--force', action='store_true', help='Force re-extraction even if cached files exist.')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-valid-samples', type=int, default=None)
    parser.add_argument('--max-test-samples', type=int, default=None)
    return parser.parse_args()


def _can_reuse_existing(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        data = np.load(path)
        rows = int(data['labels'].shape[0])
        return rows == int(expected_rows)
    except Exception:
        return False


def _extract_split(
    split: str,
    model,
    cfg,
    model_type: str,
    batch_size: int,
    image_size: int,
    feature_dir: Path,
    max_samples: int | None,
    cache_images: bool,
    reuse_existing: bool,
    force: bool,
    rgb_input_only: bool = False,
    class_rgb_stats: dict | None = None,
) -> dict:
    split_data = load_split_dataframe(split, cfg, max_samples=max_samples)
    output_path = feature_dir / f'{split}_features.npz'

    if not force and reuse_existing and _can_reuse_existing(output_path, expected_rows=split_data.available_rows):
        LOGGER.info('[%s] Reusing cached features: %s', split, output_path)
        data = np.load(output_path)
        return {
            'split': split,
            'rows': int(data['labels'].shape[0]),
            'feature_shape': list(data['features'].shape),
            'output': str(output_path),
            'resolved_image_root': str(split_data.image_root),
            'reused': True,
        }

    sequence = create_split_sequence(
        split_data=split_data,
        config=cfg,
        model_type=model_type,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=cache_images,
        rgb_input_only=rgb_input_only,
        auxiliary_quality=False,
        class_rgb_stats=class_rgb_stats,
    )

    features = model.predict(sequence, verbose=1)
    labels = sequence.get_labels()
    paths = np.array(sequence.get_paths())
    identity_features = extract_identity_features_for_paths(
        paths=paths.tolist(),
        spec=IDENTITY_SPEC,
        face_crop_expand=cfg.face_crop_expand,
    )
    combined_features = np.concatenate([features.astype(np.float32), identity_features], axis=1)

    np.savez_compressed(
        output_path,
        features=combined_features,
        cnn_features=features.astype(np.float32),
        identity_features=identity_features,
        labels=labels,
        paths=paths,
    )

    LOGGER.info(
        '[%s] Saved combined features to %s with shape=%s (cnn=%s identity=%s)',
        split,
        output_path,
        combined_features.shape,
        features.shape,
        identity_features.shape,
    )
    return {
        'split': split,
        'rows': int(labels.shape[0]),
        'feature_shape': list(combined_features.shape),
        'cnn_feature_shape': list(features.shape),
        'identity_feature_shape': list(identity_features.shape),
        'output': str(output_path),
        'resolved_image_root': str(split_data.image_root),
        'reused': False,
    }


def main() -> None:
    args = parse_args()
    cfg = get_config()

    metadata = load_optional_json(cfg.cnn_metadata_path) or {}
    model_type = normalize_model_type(str(metadata.get('backbone_name', metadata.get('model_type', args.model_type))))
    dual = bool(metadata.get('dual_hybrid_backbone', False))
    class_rgb_stats = None
    stats_path = Path(metadata.get('class_rgb_stats_path', cfg.cnn_train_rgb_stats_path))
    if dual and stats_path.exists():
        try:
            class_rgb_stats = load_joblib(stats_path)
        except OSError:
            class_rgb_stats = None

    feature_dir = Path(args.feature_dir).resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)

    cnn_model = load_cnn_model(Path(args.model_path).resolve())
    extractor = get_feature_extractor(cnn_model)
    image_size = get_model_input_size(cnn_model)

    summaries = {}
    split_limits = {
        'train': args.max_train_samples,
        'valid': args.max_valid_samples,
        'test': args.max_test_samples,
    }

    for split in ('train', 'valid', 'test'):
        summaries[split] = _extract_split(
            split=split,
            model=extractor,
            cfg=cfg,
            model_type=model_type,
            batch_size=args.batch_size,
            image_size=image_size,
            feature_dir=feature_dir,
            max_samples=split_limits[split],
            cache_images=args.cache_images,
            reuse_existing=args.reuse_existing,
            force=args.force,
            rgb_input_only=dual,
            class_rgb_stats=class_rgb_stats,
        )

    config_payload = {
        'model_type': model_type,
        'backbone_name': model_type,
        'dual_hybrid_backbone': dual,
        'image_size': image_size,
        'embedding_dim': int(extractor.output_shape[-1]),
        'identity_feature_dim': int(summaries['train']['identity_feature_shape'][1]),
        'feature_pipeline': 'cnn_embedding_plus_face_identity_descriptor',
        'class_mapping': {'0': 'Fake', '1': 'Real'},
        'feature_files': summaries,
    }

    extractor_config_path = Path(args.extractor_config).resolve()
    extractor_config_path.parent.mkdir(parents=True, exist_ok=True)
    with extractor_config_path.open('w', encoding='utf-8') as fp:
        json.dump(config_payload, fp, indent=2)

    LOGGER.info('Saved extractor config to %s', extractor_config_path)


if __name__ == '__main__':
    main()
