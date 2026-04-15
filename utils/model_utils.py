from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

# Keras 3: use PyTorch backend when TensorFlow is unavailable (e.g. Python 3.14+).
_hdfd_backend = os.environ.get('HDFD_KERAS_BACKEND')
if _hdfd_backend:
    os.environ['KERAS_BACKEND'] = _hdfd_backend
elif 'KERAS_BACKEND' not in os.environ:
    os.environ['KERAS_BACKEND'] = 'torch'


def _workspace_keras_home() -> Path:
    path = Path(__file__).resolve().parents[1] / '.keras'
    path.mkdir(parents=True, exist_ok=True)
    return path


_KERAS_HOME = _workspace_keras_home()
os.environ.setdefault('KERAS_HOME', str(_KERAS_HOME))
os.environ.setdefault('TFHUB_CACHE_DIR', str(_KERAS_HOME / 'tfhub'))

import joblib
import keras
import numpy as np
from keras import Model
from keras import metrics as keras_metrics
from keras import ops
from keras.layers import BatchNormalization, Concatenate, Dense, Dropout, GlobalAveragePooling2D, Input, Lambda
from keras.losses import Loss
from keras.optimizers import Adam
from keras.regularizers import l2
from PIL import Image

LOGGER = logging.getLogger(__name__)
ModelType = Literal['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16']


def normalize_model_type(model_type: str) -> ModelType:
    model = model_type.lower().strip()
    supported = {'efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16'}
    if model not in supported:
        raise ValueError(f"Unsupported model_type. Expected one of: {sorted(supported)}")
    return model  # type: ignore[return-value]


def default_tail_layers(model_type: str) -> int:
    model = normalize_model_type(model_type)
    if model == 'efficientnetb1':
        return 50
    if model == 'efficientnetb0':
        return 40
    if model == 'resnet50':
        return 18
    if model == 'mobilenetv2':
        return 30
    return 8


def default_deep_tail_layers(model_type: str) -> int:
    model = normalize_model_type(model_type)
    if model == 'efficientnetb1':
        return 120
    if model == 'efficientnetb0':
        return 100
    if model == 'resnet50':
        return 60
    if model == 'mobilenetv2':
        return 80
    return 20


def default_image_size(model_type: str) -> int:
    model = normalize_model_type(model_type)
    if model == 'efficientnetb1':
        return 240
    if model == 'mobilenetv2':
        return 192
    return 224


def set_seed(seed: int = 42) -> None:
    keras.utils.set_random_seed(seed)


class BinaryFocalLoss(Loss):
    """Binary focal loss with optional label smoothing on targets (reduces overconfidence)."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        # Prevent accidental broadcast (batch,) vs (batch,1) under torch backend.
        y_true = ops.cast(y_true, y_pred.dtype)
        y_true = ops.reshape(y_true, (-1, 1))
        y_pred = ops.reshape(y_pred, (-1, 1))
        y_pred = ops.clip(y_pred, 1e-7, 1.0 - 1e-7)
        ls = self.label_smoothing
        if ls > 0.0:
            y_true = y_true * (1.0 - ls) + 0.5 * ls
        bce = -(y_true * ops.log(y_pred) + (1.0 - y_true) * ops.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal = alpha_t * ops.power(1.0 - p_t, self.gamma) * bce
        return ops.reshape(focal, (-1,))

    def get_config(self):
        base = super().get_config()
        base.update(
            {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'label_smoothing': self.label_smoothing,
            }
        )
        return base


def predict_cnn_real_probs(model: Model, data, verbose: int = 0) -> np.ndarray:
    """First output tensor is always P(real); supports legacy single-output and multi-output CNNs."""
    out = model.predict(data, verbose=verbose)
    if isinstance(out, (list, tuple)):
        return np.asarray(out[0], dtype=np.float32).reshape(-1)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def get_preprocess_function(model_type: str):
    model = normalize_model_type(model_type)
    if model == 'resnet50':
        from keras.applications.resnet50 import preprocess_input

        return preprocess_input
    if model == 'vgg16':
        from keras.applications.vgg16 import preprocess_input

        return preprocess_input
    if model == 'efficientnetb0':
        from keras.applications.efficientnet import preprocess_input

        return preprocess_input
    if model == 'efficientnetb1':
        from keras.applications.efficientnet import preprocess_input

        return preprocess_input
    from keras.applications.mobilenet_v2 import preprocess_input

    return preprocess_input


def _build_backbone_with_fallback(factory, model_name: str, input_shape: tuple[int, int, int]):
    try:
        return factory(weights='imagenet', include_top=False, input_shape=input_shape), True
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning(
            'Could not load ImageNet weights for %s (%s). Falling back to random initialization.',
            model_name,
            exc,
        )
        return factory(weights=None, include_top=False, input_shape=input_shape), False


def _build_backbone(model_type: str, image_size: int) -> tuple[Model, bool]:
    model = normalize_model_type(model_type)
    input_shape = (image_size, image_size, 3)

    if model == 'resnet50':
        from keras.applications import ResNet50

        return _build_backbone_with_fallback(ResNet50, 'ResNet50', input_shape)

    if model == 'vgg16':
        from keras.applications import VGG16

        return _build_backbone_with_fallback(VGG16, 'VGG16', input_shape)

    if model == 'efficientnetb0':
        from keras.applications import EfficientNetB0

        return _build_backbone_with_fallback(EfficientNetB0, 'EfficientNetB0', input_shape)

    if model == 'efficientnetb1':
        from keras.applications import EfficientNetB1

        return _build_backbone_with_fallback(EfficientNetB1, 'EfficientNetB1', input_shape)

    from keras.applications import MobileNetV2

    return _build_backbone_with_fallback(MobileNetV2, 'MobileNetV2', input_shape)


def build_cnn_model(
    model_type: str = 'efficientnetb0',
    image_size: int = 224,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.4,
    label_smoothing: float = 0.1,
) -> Model:
    model_type = normalize_model_type(model_type)
    backbone, pretrained = _build_backbone(model_type, image_size)
    if pretrained:
        LOGGER.info('Using pretrained %s backbone weights.', model_type)

    x = backbone.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='embedding_bn')(x)
    x = Dense(
        128,
        activation='relu',
        name='embedding',
        kernel_regularizer=l2(1e-4),
        bias_regularizer=l2(1e-4),
    )(x)
    x = Dropout(dropout_rate, name='dropout')(x)
    output = Dense(1, activation='sigmoid', name='prediction', dtype='float32')(x)

    model = Model(inputs=backbone.input, outputs=output, name=f'{model_type}_deepfake_detector')
    setattr(model, '_hdfd_pretrained_backbone', bool(pretrained))
    setattr(model, '_hdfd_dual_backbone', False)
    compile_cnn_model(
        model=model,
        learning_rate=learning_rate,
        label_smoothing=label_smoothing,
    )
    return model


def build_hybrid_dual_backbone_cnn(
    image_size: int = 224,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.4,
    l2_dense: float = 1e-4,
    aux_loss_weight: float = 0.25,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
) -> Model:
    """EfficientNetB0 + MobileNetV2 on shared RGB input (per-branch ImageNet preprocess), concat → XGBoost features."""
    from keras.applications.efficientnet import preprocess_input as eff_preprocess
    from keras.applications.mobilenet_v2 import preprocess_input as mob_preprocess

    eff_backbone, pre_e = _build_backbone('efficientnetb0', image_size)
    mob_backbone, pre_m = _build_backbone('mobilenetv2', image_size)
    eff_backbone.name = 'eff_backbone'
    mob_backbone.name = 'mob_backbone'

    inputs = Input(shape=(image_size, image_size, 3), name='rgb_input')
    eff_in = Lambda(lambda x: eff_preprocess(x), name='eff_preprocess')(inputs)
    mob_in = Lambda(lambda x: mob_preprocess(x), name='mob_preprocess')(inputs)

    e = eff_backbone(eff_in)
    m = mob_backbone(mob_in)
    e = GlobalAveragePooling2D(name='eff_gap')(e)
    m = GlobalAveragePooling2D(name='mob_gap')(m)
    merged = Concatenate(name='hybrid_concat')([e, m])
    merged = BatchNormalization(name='hybrid_bn')(merged)
    emb = Dense(
        256,
        activation='relu',
        name='hybrid_embedding',
        kernel_regularizer=l2(l2_dense),
        bias_regularizer=l2(l2_dense),
    )(merged)
    emb = Dropout(dropout_rate, name='hybrid_dropout')(emb)
    main_out = Dense(1, activation='sigmoid', name='prediction', dtype='float32')(emb)
    aux_out = Dense(1, activation='sigmoid', name='quality_score', dtype='float32')(emb)

    model = Model(
        inputs=inputs,
        outputs={'prediction': main_out, 'quality_score': aux_out},
        name='hybrid_dual_efficientnet_mobilenet',
    )
    setattr(model, '_hdfd_pretrained_backbone', bool(pre_e and pre_m))
    setattr(model, '_hdfd_dual_backbone', True)
    setattr(model, '_hdfd_aux_loss_weight', float(aux_loss_weight))

    compile_dual_hybrid_cnn_model(
        model=model,
        learning_rate=learning_rate,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
        aux_loss_weight=aux_loss_weight,
    )
    return model


def compile_cnn_model(
    model: Model,
    learning_rate: float,
    label_smoothing: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> None:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing),
        metrics=[
            keras_metrics.BinaryAccuracy(name='accuracy'),
            keras_metrics.AUC(name='auc'),
            keras_metrics.Precision(name='precision'),
            keras_metrics.Recall(name='recall'),
        ],
    )


def compile_dual_hybrid_cnn_model(
    model: Model,
    learning_rate: float,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1,
    aux_loss_weight: float = 0.25,
) -> None:
    aux_w = float(getattr(model, '_hdfd_aux_loss_weight', aux_loss_weight))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'prediction': BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, label_smoothing=label_smoothing),
            'quality_score': 'mse',
        },
        loss_weights={'prediction': 1.0, 'quality_score': aux_w},
        metrics={
            'prediction': [
                keras_metrics.BinaryAccuracy(name='accuracy'),
                keras_metrics.AUC(name='auc'),
                keras_metrics.Precision(name='precision'),
                keras_metrics.Recall(name='recall'),
            ],
            'quality_score': [keras_metrics.MeanAbsoluteError(name='mae')],
        },
    )


def _base_layers(model: Model) -> list:
    gap_idx = next((idx for idx, layer in enumerate(model.layers) if layer.name == 'global_avg_pool'), None)
    if gap_idx is None:
        raise ValueError('global_avg_pool layer not found. Unexpected model architecture.')
    return model.layers[:gap_idx]


def set_trainable_for_warmup(model: Model) -> None:
    base = _base_layers(model)
    for layer in base:
        layer.trainable = False
    for layer in model.layers[len(base) :]:
        layer.trainable = True


def set_trainable_for_finetune(model: Model, tail_layers: int) -> None:
    base = _base_layers(model)
    tail_layers = max(1, min(tail_layers, len(base)))

    for layer in base[:-tail_layers]:
        layer.trainable = False
    for layer in base[-tail_layers:]:
        layer.trainable = True

    for layer in model.layers[len(base) :]:
        layer.trainable = True


def set_trainable_backbone_last_n(model: Model, n: int) -> None:
    """Unfreeze only the last n layers of the convolutional backbone (below global_avg_pool)."""
    base = _base_layers(model)
    n = max(1, min(int(n), len(base)))
    for layer in base[:-n]:
        layer.trainable = False
    for layer in base[-n:]:
        layer.trainable = True
    for layer in model.layers[len(base) :]:
        layer.trainable = True


def set_trainable_for_full(model: Model) -> None:
    for layer in model.layers:
        layer.trainable = True


def is_dual_backbone_model(model: Model) -> bool:
    return bool(getattr(model, '_hdfd_dual_backbone', False))


def set_trainable_for_warmup_dual(model: Model) -> None:
    eff = model.get_layer('eff_backbone')
    mob = model.get_layer('mob_backbone')
    eff.trainable = False
    mob.trainable = False
    for layer in model.layers:
        if layer.name not in {'eff_backbone', 'mob_backbone'}:
            layer.trainable = True


def set_trainable_backbone_last_n_dual(model: Model, n_eff: int, n_mob: int) -> None:
    eff = model.get_layer('eff_backbone')
    mob = model.get_layer('mob_backbone')
    eff.trainable = True
    mob.trainable = True
    for layer in eff.layers:
        layer.trainable = False
    for layer in mob.layers:
        layer.trainable = False
    n_eff = max(1, min(int(n_eff), len(eff.layers)))
    n_mob = max(1, min(int(n_mob), len(mob.layers)))
    for layer in eff.layers[-n_eff:]:
        layer.trainable = True
    for layer in mob.layers[-n_mob:]:
        layer.trainable = True
    for layer in model.layers:
        if layer.name not in {'eff_backbone', 'mob_backbone'}:
            layer.trainable = True


def set_trainable_for_full_dual(model: Model) -> None:
    for layer in model.layers:
        layer.trainable = True


def get_feature_extractor(cnn_model: Model) -> Model:
    for name in ('hybrid_embedding', 'embedding'):
        try:
            embedding_layer = cnn_model.get_layer(name)
            return Model(inputs=cnn_model.input, outputs=embedding_layer.output, name='cnn_feature_extractor')
        except ValueError:
            continue
    raise ValueError('Could not find hybrid_embedding or embedding layer on CNN model.')


def get_model_input_size(cnn_model: Model) -> int:
    input_shape = cnn_model.input_shape
    if not isinstance(input_shape, tuple) or len(input_shape) < 3 or input_shape[1] is None:
        raise ValueError(f'Unable to infer model input size from shape: {input_shape}')
    return int(input_shape[1])


def save_keras_model(model: Model, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def load_cnn_model(model_path: Path) -> Model:
    if not model_path.exists():
        raise FileNotFoundError(f'CNN model not found at: {model_path}')
    custom_objects = {'BinaryFocalLoss': BinaryFocalLoss}
    try:
        return keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
    except Exception as first_exc:  # pylint: disable=broad-except
        LOGGER.warning(
            'load_model(compile=False) failed for %s (%s); retrying with compile=True.',
            model_path,
            first_exc,
        )
        return keras.models.load_model(str(model_path), custom_objects=custom_objects)


def save_xgb_model(model, model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_xgb_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f'XGBoost model not found at: {model_path}')
    return joblib.load(model_path)


def save_joblib(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    return joblib.load(path)


def load_optional_joblib(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def save_json_file(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)
    return path


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'JSON file not found: {path}')
    # Accept UTF-8 with/without BOM so PowerShell-written JSON is parsed reliably.
    with path.open('r', encoding='utf-8-sig') as fp:
        return json.load(fp)


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return load_json_file(path)
    except Exception:
        return None


def preprocess_single_image_with_meta(
    image: Image.Image,
    image_size: int,
    model_type: str,
    use_face_crop: bool | None = None,
    face_crop_expand: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Face crop (Haar) + 224 resize + ImageNet preprocess for EfficientNet-style heads."""
    from utils.config import get_config
    from utils.face_detection import extract_face_or_fallback, resize_rgb

    cfg = get_config()
    ufc = cfg.use_face_crop if use_face_crop is None else use_face_crop
    exp = cfg.face_crop_expand if face_crop_expand is None else face_crop_expand

    if ufc:
        cropped, meta = extract_face_or_fallback(image, expand=float(exp))
        resized = resize_rgb(cropped, image_size)
    else:
        meta = {'face_detected': False, 'method': 'full_resize', 'reason': 'face_crop_disabled'}
        resized = image.convert('RGB').resize((image_size, image_size), Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32)
    cnn_meta = load_optional_json(cfg.cnn_metadata_path) or {}
    if bool(cnn_meta.get('dual_hybrid_backbone')):
        arr = np.clip(arr, 0.0, 255.0)
        return np.expand_dims(arr, axis=0), meta
    preprocess_fn = get_preprocess_function(model_type)
    arr = preprocess_fn(arr)
    return np.expand_dims(arr, axis=0), meta


def preprocess_single_image(image: Image.Image, image_size: int, model_type: str) -> np.ndarray:
    batch, _meta = preprocess_single_image_with_meta(image, image_size, model_type)
    return batch


def probability_to_prediction(
    prob_real: float,
    threshold: float = 0.5,
    uncertain_low: float | None = None,
    uncertain_high: float | None = None,
) -> dict[str, float | str]:
    threshold = float(threshold)
    lo = uncertain_low if uncertain_low is not None else None
    hi = uncertain_high if uncertain_high is not None else None
    if lo is not None and hi is not None and float(lo) < float(prob_real) < float(hi):
        return {
            'predicted_class': 'Uncertain',
            'confidence': float(max(0.0, 1.0 - abs(float(prob_real) - 0.5) * 2.0)),
            'probabilities': {
                'real': float(prob_real),
                'fake': float(1.0 - prob_real),
            },
            'decision_threshold': threshold,
            'uncertainty_band': [float(lo), float(hi)],
        }
    predicted_class = 'Real' if prob_real >= threshold else 'Fake'
    confidence = prob_real if predicted_class == 'Real' else 1.0 - prob_real
    return {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            'real': float(prob_real),
            'fake': float(1.0 - prob_real),
        },
        'decision_threshold': threshold,
    }
