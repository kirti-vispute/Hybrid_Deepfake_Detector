from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _as_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


@dataclass
class AppConfig:
    # Defaults are relative to the repo root so clones work on any machine.
    # Override with HDFD_* environment variables or absolute paths if needed.
    dataset_root: Path = _as_path('data')
    # Image files live under dataset/real and dataset/fake (paths in CSVs are real/... or fake/...).
    image_root: Path = _as_path('dataset')
    train_csv: Path = _as_path('data/train.csv')
    valid_csv: Path = _as_path('data/valid.csv')
    test_csv: Path = _as_path('data/test.csv')

    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    backbone_name: str = 'resnet50'
    model_type: str = 'resnet50'

    # Mode-specific backbone defaults (ResNet-50 as primary transfer-learning baseline).
    fast_backbone_name: str = 'resnet50'
    strong_backbone_name: str = 'resnet50'

    warmup_epochs: int = 1
    finetune_epochs: int = 4
    dropout_rate: float = 0.4
    label_smoothing: float = 0.1

    # Keep a single primary backbone by default (ResNet-50).
    use_dual_hybrid_cnn: bool = False
    equalize_ycrcb: bool = True
    match_fake_to_real_stats: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    aux_loss_weight: float = 0.25
    hybrid_fpr_retrain_threshold: float = 0.20
    max_hard_negative_rounds: int = 2

    fast_mode: bool = True
    rigorous_mode: bool = False
    epochs_head: int = 3
    epochs_finetune_stage2: int = 5
    epochs_finetune_stage3: int = 3
    use_mixed_precision: bool = True
    use_class_weights: bool = True
    dataloader_workers: int = 2
    cache_images: bool = False
    max_cache_images: int = 1024

    # Fast-iteration defaults for local experimentation.
    fast_max_train_samples: int = 30000
    fast_max_valid_samples: int = 6000

    backend_port: int = 8000
    frontend_port: int = 5173
    max_upload_size_mb: int = 8
    max_missing_ratio: float = 0.30

    # Face crop (Haar) before resize — matches inference and improves real-world photos.
    use_face_crop: bool = True
    face_crop_expand: float = 1.25

    # Hybrid output: mark borderline scores as UNCERTAIN (calibrated P(real)).
    uncertain_prob_low: float = 0.4
    uncertain_prob_high: float = 0.6

    aug_hflip_prob: float = 0.5
    aug_brightness_delta: float = 18.0
    aug_contrast_low: float = 0.90
    aug_contrast_high: float = 1.10

    artifacts_dir: Path = _as_path('artifacts')
    results_dir: Path = _as_path('results')
    models_dir: Path = _as_path('models')

    cnn_model_path: Path = _as_path('models/cnn_baseline.keras')
    xgb_model_path: Path = _as_path('models/xgboost_hybrid.joblib')
    xgb_scaler_path: Path = _as_path('models/hybrid_feature_scaler.joblib')
    hybrid_pca_path: Path = _as_path('models/hybrid_pca.joblib')
    cnn_calibrator_path: Path = _as_path('models/cnn_calibrator.joblib')
    hybrid_calibrator_path: Path = _as_path('models/hybrid_calibrator.joblib')
    smart_router_path: Path = _as_path('models/smart_router.json')
    cnn_metadata_path: Path = _as_path('models/cnn_metadata.json')
    cnn_train_rgb_stats_path: Path = _as_path('models/cnn_train_rgb_stats.joblib')
    hybrid_metadata_path: Path = _as_path('models/hybrid_metadata.json')
    classical_model_path: Path = _as_path('models/classical_fallback_model.joblib')
    classical_metadata_path: Path = _as_path('models/classical_fallback_metadata.json')
    classical_feature_dir: Path = _as_path('artifacts/classical_features')
    # Written by evaluate.py: selects cnn_direct vs hybrid for production inference.
    production_inference_path: Path = _as_path('models/production_inference.json')

    feature_dir: Path = _as_path('artifacts/features')
    extractor_config_path: Path = _as_path('artifacts/feature_extractor_config.json')
    history_json_path: Path = _as_path('results/cnn_training_history.json')
    training_curve_path: Path = _as_path('results/cnn_training_curves.png')
    dataset_audit_path: Path = _as_path('results/dataset_audit.json')

    random_seed: int = 42

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_output_dirs(self) -> None:
        for path in (
            self.artifacts_dir,
            self.results_dir,
            self.models_dir,
            self.feature_dir,
            self.cnn_model_path.parent,
            self.xgb_model_path.parent,
            self.xgb_scaler_path.parent,
            self.cnn_calibrator_path.parent,
            self.hybrid_calibrator_path.parent,
            self.hybrid_pca_path.parent,
            self.smart_router_path.parent,
            self.extractor_config_path.parent,
            self.dataset_audit_path.parent,
            self.production_inference_path.parent,
            self.classical_model_path.parent,
            self.classical_metadata_path.parent,
            self.classical_feature_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _env_override(cfg: AppConfig) -> AppConfig:
    env_map: dict[str, tuple[str, callable]] = {
        'HDFD_DATASET_ROOT': ('dataset_root', _as_path),
        'HDFD_IMAGE_ROOT': ('image_root', _as_path),
        'HDFD_TRAIN_CSV': ('train_csv', _as_path),
        'HDFD_VALID_CSV': ('valid_csv', _as_path),
        'HDFD_TEST_CSV': ('test_csv', _as_path),
        'HDFD_IMAGE_SIZE': ('image_size', int),
        'HDFD_BATCH_SIZE': ('batch_size', int),
        'HDFD_EPOCHS': ('epochs', int),
        'HDFD_LEARNING_RATE': ('learning_rate', float),
        'HDFD_MODEL_TYPE': ('model_type', str),
        'HDFD_BACKBONE_NAME': ('backbone_name', str),
        'HDFD_FAST_BACKBONE_NAME': ('fast_backbone_name', str),
        'HDFD_STRONG_BACKBONE_NAME': ('strong_backbone_name', str),
        'HDFD_BACKEND_PORT': ('backend_port', int),
        'HDFD_FRONTEND_PORT': ('frontend_port', int),
        'HDFD_MAX_UPLOAD_MB': ('max_upload_size_mb', int),
        'HDFD_FAST_MODE': ('fast_mode', _as_bool),
        'HDFD_RIGOROUS_MODE': ('rigorous_mode', _as_bool),
        'HDFD_EPOCHS_HEAD': ('epochs_head', int),
        'HDFD_EPOCHS_FINETUNE_STAGE2': ('epochs_finetune_stage2', int),
        'HDFD_EPOCHS_FINETUNE_STAGE3': ('epochs_finetune_stage3', int),
        'HDFD_USE_MIXED_PRECISION': ('use_mixed_precision', _as_bool),
        'HDFD_USE_CLASS_WEIGHTS': ('use_class_weights', _as_bool),
        'HDFD_DATALOADER_WORKERS': ('dataloader_workers', int),
        'HDFD_CACHE_IMAGES': ('cache_images', _as_bool),
        'HDFD_MAX_CACHE_IMAGES': ('max_cache_images', int),
        'HDFD_FAST_MAX_TRAIN_SAMPLES': ('fast_max_train_samples', int),
        'HDFD_FAST_MAX_VALID_SAMPLES': ('fast_max_valid_samples', int),
        'HDFD_CNN_MODEL_PATH': ('cnn_model_path', _as_path),
        'HDFD_XGB_MODEL_PATH': ('xgb_model_path', _as_path),
        'HDFD_XGB_SCALER_PATH': ('xgb_scaler_path', _as_path),
        'HDFD_HYBRID_PCA_PATH': ('hybrid_pca_path', _as_path),
        'HDFD_CNN_CALIBRATOR_PATH': ('cnn_calibrator_path', _as_path),
        'HDFD_HYBRID_CALIBRATOR_PATH': ('hybrid_calibrator_path', _as_path),
        'HDFD_SMART_ROUTER_PATH': ('smart_router_path', _as_path),
        'HDFD_PRODUCTION_INFERENCE_PATH': ('production_inference_path', _as_path),
        'HDFD_CLASSICAL_MODEL_PATH': ('classical_model_path', _as_path),
        'HDFD_CLASSICAL_METADATA_PATH': ('classical_metadata_path', _as_path),
        'HDFD_USE_FACE_CROP': ('use_face_crop', _as_bool),
        'HDFD_FACE_CROP_EXPAND': ('face_crop_expand', float),
    }
    for env_name, (field_name, caster) in env_map.items():
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        setattr(cfg, field_name, caster(raw_value))
    return cfg


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    cfg = AppConfig()
    cfg = _env_override(cfg)
    if not cfg.model_type:
        cfg.model_type = cfg.backbone_name
    if not cfg.backbone_name:
        cfg.backbone_name = cfg.model_type
    cfg.ensure_output_dirs()
    return cfg


def reset_config_cache() -> None:
    get_config.cache_clear()
