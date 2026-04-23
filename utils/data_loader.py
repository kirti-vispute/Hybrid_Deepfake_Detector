from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from keras.utils import Sequence

from utils.config import AppConfig
from utils.face_detection import extract_face_or_fallback, resize_rgb
from utils.model_utils import get_preprocess_function
from utils.path_utils import resolve_absolute_image_path, resolve_existing_image_root

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {'path', 'label', 'label_str'}
SPLIT_TO_CSV_FIELD = {
    'train': 'train_csv',
    'valid': 'valid_csv',
    'test': 'test_csv',
}


def laplacian_sharpness_score_01(rgb_float: np.ndarray) -> float:
    """Normalized sharpness proxy in [0, 1] from Laplacian variance (RGB float 0–255)."""
    u8 = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = float(lap.var())
    return float(np.clip(np.log1p(v) / 8.0, 0.0, 1.0))


def equalize_ycrcb_rgb(rgb_float: np.ndarray) -> np.ndarray:
    """Reduce lighting bias via Y-channel histogram equalization."""
    u8 = np.clip(rgb_float, 0.0, 255.0).astype(np.uint8)
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge((y_eq, cr, cb))
    rgb = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return rgb.astype(np.float32)


def compute_class_rgb_mean_std(
    dataframe: pd.DataFrame,
    image_size: int,
    use_face_crop: bool,
    face_crop_expand: float,
    max_samples_per_class: int = 2048,
    seed: int = 42,
) -> dict[int, dict[str, np.ndarray]]:
    """Dataset-level RGB mean/std per class (0=fake, 1=real) for distribution matching."""
    rng = np.random.default_rng(seed)
    out: dict[int, dict[str, np.ndarray]] = {}
    for cls in (0, 1):
        sub = dataframe.loc[dataframe['label'].astype(int) == cls]
        if len(sub) == 0:
            continue
        idx = np.arange(len(sub))
        if len(idx) > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)
        rows = sub.iloc[idx]
        acc: list[np.ndarray] = []
        for _, row in rows.iterrows():
            p = Path(row['abs_path'])
            try:
                with Image.open(p) as img:
                    rgb = img.convert('RGB')
                    if use_face_crop:
                        cropped, _ = extract_face_or_fallback(rgb, expand=float(face_crop_expand))
                        rgb = resize_rgb(cropped, image_size)
                    else:
                        rgb = rgb.resize((image_size, image_size), Image.BILINEAR)
                    acc.append(np.asarray(rgb, dtype=np.float32))
            except (UnidentifiedImageError, OSError, ValueError):
                continue
        if not acc:
            continue
        stacked = np.stack(acc, axis=0)
        mean = stacked.mean(axis=(0, 1, 2))
        std = stacked.std(axis=(0, 1, 2))
        std = np.maximum(std, 1.0)
        out[cls] = {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}
    return out


@dataclass
class SplitData:
    split: str
    dataframe: pd.DataFrame
    image_root: Path
    total_rows: int
    available_rows: int
    missing_rows: int
    label_distribution: dict[str, int]
    label_folder_mismatches: int


def _infer_label_from_path(relative_path: str) -> int | None:
    parts = str(relative_path).replace('\\', '/').lower().split('/')
    if 'real' in parts:
        return 1
    if 'fake' in parts:
        return 0
    return None


class CSVImageSequence(Sequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        batch_size: int,
        image_size: int,
        model_type: str,
        training: bool,
        shuffle: bool,
        seed: int = 42,
        aug_hflip_prob: float = 0.5,
        aug_brightness_delta: float = 18.0,
        aug_contrast_low: float = 0.90,
        aug_contrast_high: float = 1.10,
        cache_images: bool = False,
        max_cache_images: int = 1024,
        use_face_crop: bool = True,
        face_crop_expand: float = 1.25,
        rgb_input_only: bool = False,
        auxiliary_quality: bool = False,
        class_rgb_stats: dict[int, dict[str, np.ndarray]] | None = None,
        equalize_ycrcb: bool = False,
        match_fake_to_real_stats: bool = False,
    ) -> None:
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.batch_size = max(1, int(batch_size))
        self.image_size = int(image_size)
        self.model_type = model_type
        self.training = training
        self.shuffle = shuffle
        self.use_face_crop = bool(use_face_crop)
        self.face_crop_expand = float(face_crop_expand)
        self.rgb_input_only = bool(rgb_input_only)
        self.auxiliary_quality = bool(auxiliary_quality)
        self.class_rgb_stats = class_rgb_stats
        self.equalize_ycrcb = bool(equalize_ycrcb)
        self.match_fake_to_real_stats = bool(match_fake_to_real_stats)
        self.preprocess_fn = get_preprocess_function(model_type)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(self.df))

        self.aug_hflip_prob = float(aug_hflip_prob)
        self.aug_brightness_delta = float(aug_brightness_delta)
        self.aug_contrast_low = float(aug_contrast_low)
        self.aug_contrast_high = float(aug_contrast_high)

        self.cache_images = bool(cache_images)
        self.max_cache_images = max(0, int(max_cache_images))
        self._raw_cache: dict[str, np.ndarray] = {}
        self._cache_order: list[str] = []

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _augment(self, image_array: np.ndarray) -> np.ndarray:
        img = image_array

        if self.rng.random() < self.aug_hflip_prob:
            img = np.fliplr(img)

        if self.rng.random() < 0.30:
            delta = self.rng.uniform(-self.aug_brightness_delta, self.aug_brightness_delta)
            img = np.clip(img + delta, 0.0, 255.0)

        if self.rng.random() < 0.30:
            contrast = self.rng.uniform(self.aug_contrast_low, self.aug_contrast_high)
            mean = img.mean(axis=(0, 1), keepdims=True)
            img = np.clip((img - mean) * contrast + mean, 0.0, 255.0)

        if self.rng.random() < 0.25:
            h, w = img.shape[:2]
            z = float(self.rng.uniform(0.88, 1.0))
            nh, nw = max(8, int(h * z)), max(8, int(w * z))
            if nh < h and nw < w:
                sy = int(self.rng.integers(0, h - nh + 1))
                sx = int(self.rng.integers(0, w - nw + 1))
                crop = img[sy : sy + nh, sx : sx + nw]
                pil = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
                pil = pil.resize((w, h), Image.BILINEAR)
                img = np.asarray(pil, dtype=np.float32)

        if self.rng.random() < 0.20:
            sigma = float(self.rng.uniform(2.0, 6.0))
            img = np.clip(img + self.rng.normal(0.0, sigma, img.shape), 0.0, 255.0)

        return img.astype(np.float32, copy=False)

    def _apply_distribution_matching(self, raw: np.ndarray, label: int) -> np.ndarray:
        if not self.match_fake_to_real_stats or self.class_rgb_stats is None:
            return raw
        if int(label) != 0:
            return raw
        sf = self.class_rgb_stats.get(0)
        sr = self.class_rgb_stats.get(1)
        if sf is None or sr is None:
            return raw
        mean_f, std_f = sf['mean'], sf['std']
        mean_r, std_r = sr['mean'], sr['std']
        x = (raw - mean_f) / (std_f + 1e-6)
        return x * std_r + mean_r

    def _load_raw_image(self, abs_path: Path) -> np.ndarray:
        try:
            with Image.open(abs_path) as img:
                rgb = img.convert('RGB')
                if self.use_face_crop:
                    cropped, _meta = extract_face_or_fallback(rgb, expand=self.face_crop_expand)
                    rgb = resize_rgb(cropped, self.image_size)
                else:
                    rgb = rgb.resize((self.image_size, self.image_size), Image.BILINEAR)
                arr = np.asarray(rgb, dtype=np.float32)
                return arr
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            LOGGER.warning('Image read failed for %s (%s). Using zero-image fallback.', abs_path, exc)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

    def _get_cached_or_load_raw(self, abs_path: Path) -> np.ndarray:
        key = str(abs_path)
        if self.cache_images and key in self._raw_cache:
            return self._raw_cache[key].copy()

        arr = self._load_raw_image(abs_path)

        if self.cache_images and self.max_cache_images > 0:
            if key not in self._raw_cache:
                self._raw_cache[key] = arr
                self._cache_order.append(key)
                if len(self._cache_order) > self.max_cache_images:
                    oldest = self._cache_order.pop(0)
                    self._raw_cache.pop(oldest, None)
        return arr.copy()

    def _load_image(self, abs_path: Path, label: int) -> np.ndarray:
        arr = self._get_cached_or_load_raw(abs_path)
        q_score = laplacian_sharpness_score_01(arr)
        arr = self._apply_distribution_matching(arr, int(label))
        if self.equalize_ycrcb:
            arr = equalize_ycrcb_rgb(arr)
        if self.training:
            arr = self._augment(arr)
        if self.rgb_input_only:
            return np.clip(arr, 0.0, 255.0).astype(np.float32), float(q_score)
        return self.preprocess_fn(arr), float(q_score)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, dict[str, np.ndarray]]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.df))
        batch_indices = self.indices[start:stop]

        images: list[np.ndarray] = []
        labels: list[float] = []
        qualities: list[float] = []

        for row_idx in batch_indices:
            row = self.df.iloc[row_idx]
            abs_path = Path(row['abs_path'])
            lbl = int(row['label'])
            img, q = self._load_image(abs_path, lbl)
            images.append(img)
            labels.append(float(lbl))
            qualities.append(q)

        x_batch = np.stack(images).astype(np.float32)
        if self.auxiliary_quality:
            y_batch = {
                'prediction': np.asarray(labels, dtype=np.float32).reshape(-1, 1),
                'quality_score': np.asarray(qualities, dtype=np.float32).reshape(-1, 1),
            }
            return x_batch, y_batch
        y_batch = np.asarray(labels, dtype=np.float32)
        return x_batch, y_batch

    def get_labels(self) -> np.ndarray:
        return self.df['label'].to_numpy(dtype=np.int32)

    def get_paths(self) -> list[str]:
        return self.df['abs_path'].astype(str).tolist()


def _read_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f'Missing required columns in {csv_path.name}: {sorted(missing_cols)}')

    df = df.copy()
    df['path'] = df['path'].astype(str).str.strip()
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['label_str'] = df['label_str'].astype(str).str.strip().str.lower()
    df = df.dropna(subset=['path', 'label'])
    df['label'] = df['label'].astype(int)

    invalid_labels = sorted(set(df['label'].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f'Unexpected labels detected in {csv_path.name}: {invalid_labels}. Expected only 0 and 1.')

    return df


def _sample_dataframe(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or max_samples >= len(df):
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _count_label_folder_mismatches(df: pd.DataFrame) -> int:
    inferred = df['path'].apply(_infer_label_from_path)
    mask = inferred.notna()
    if not mask.any():
        return 0
    inferred_labels = inferred[mask].astype(int)
    actual_labels = df.loc[mask, 'label'].astype(int)
    return int((inferred_labels != actual_labels).sum())


def load_split_dataframe(
    split: str,
    config: AppConfig,
    strict_missing_ratio: float | None = None,
    max_missing_warnings: int = 10,
    max_samples: int | None = None,
) -> SplitData:
    if split not in SPLIT_TO_CSV_FIELD:
        raise ValueError("split must be one of: 'train', 'valid', 'test'")

    csv_path = Path(getattr(config, SPLIT_TO_CSV_FIELD[split]))
    df = _read_csv(csv_path)
    rows_before_sample = int(len(df))
    if max_samples is not None:
        df = _sample_dataframe(df, max_samples=max_samples, seed=config.random_seed)
        if rows_before_sample != len(df):
            LOGGER.info(
                '[%s] Subsampled to %d rows (max_samples=%s; %d rows existed before sampling)',
                split,
                len(df),
                max_samples,
                rows_before_sample,
            )

    image_root = resolve_existing_image_root(config.image_root, df['path'].tolist())

    df['abs_path'] = df['path'].apply(lambda rel: str(resolve_absolute_image_path(config.image_root, rel)))
    exists_mask = df['abs_path'].apply(lambda p: Path(p).exists())

    missing_rows = int((~exists_mask).sum())
    total_rows = int(len(df))

    if missing_rows > 0:
        missing_examples = df.loc[~exists_mask, 'abs_path'].head(max_missing_warnings).tolist()
        LOGGER.warning('[%s] Missing files: %d/%d', split, missing_rows, total_rows)
        for sample in missing_examples:
            LOGGER.warning('[%s] Missing sample: %s', split, sample)

    available_df = df.loc[exists_mask].reset_index(drop=True)
    label_folder_mismatches = _count_label_folder_mismatches(available_df)
    if label_folder_mismatches > 0:
        LOGGER.warning('[%s] Label-path mismatches found: %d', split, label_folder_mismatches)

    available_rows = int(len(available_df))
    label_distribution = available_df['label_str'].value_counts().to_dict()

    ratio = (missing_rows / total_rows) if total_rows else 0.0
    threshold = config.max_missing_ratio if strict_missing_ratio is None else strict_missing_ratio
    if ratio > threshold:
        raise RuntimeError(
            f'[{split}] missing ratio {ratio:.2%} exceeds threshold {threshold:.2%}. '
            f'Check image_root and CSV paths.'
        )

    return SplitData(
        split=split,
        dataframe=available_df,
        image_root=image_root,
        total_rows=total_rows,
        available_rows=available_rows,
        missing_rows=missing_rows,
        label_distribution=label_distribution,
        label_folder_mismatches=label_folder_mismatches,
    )


def audit_dataset(
    config: AppConfig,
    check_corrupt: bool = False,
    max_corrupt_checks_per_split: int | None = None,
    max_log_samples: int = 20,
    output_json: Path | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {'splits': {}}

    for split in ('train', 'valid', 'test'):
        split_data = load_split_dataframe(split, config, strict_missing_ratio=config.max_missing_ratio)
        df = split_data.dataframe

        corrupt_paths: list[str] = []
        checked = 0
        if check_corrupt:
            for abs_path in df['abs_path'].tolist():
                if max_corrupt_checks_per_split is not None and checked >= max_corrupt_checks_per_split:
                    break
                checked += 1
                try:
                    with Image.open(abs_path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError, ValueError):
                    corrupt_paths.append(abs_path)

        sample_rows = df.head(5)
        sample_resolved_paths = sample_rows['abs_path'].astype(str).tolist()
        sample_relative_paths = sample_rows['path'].astype(str).tolist()

        split_summary = {
            'total_rows': split_data.total_rows,
            'available_rows': split_data.available_rows,
            'missing_rows': split_data.missing_rows,
            'label_distribution': split_data.label_distribution,
            'label_folder_mismatches': split_data.label_folder_mismatches,
            'resolved_image_root': str(split_data.image_root),
            'sample_relative_paths': sample_relative_paths,
            'sample_resolved_paths': sample_resolved_paths,
            'corrupt_checked_rows': checked,
            'corrupt_rows': len(corrupt_paths),
            'corrupt_examples': corrupt_paths[:max_log_samples],
        }

        summary['splits'][split] = split_summary

        LOGGER.info(
            '[%s] total=%d available=%d missing=%d label_mismatches=%d corrupt_checked=%d corrupt=%d',
            split,
            split_summary['total_rows'],
            split_summary['available_rows'],
            split_summary['missing_rows'],
            split_summary['label_folder_mismatches'],
            split_summary['corrupt_checked_rows'],
            split_summary['corrupt_rows'],
        )
        for path in split_summary['corrupt_examples']:
            LOGGER.warning('[%s] Corrupt image: %s', split, path)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open('w', encoding='utf-8') as fp:
            json.dump(summary, fp, indent=2)

    return summary


def validate_dataset(config: AppConfig, strict_missing_ratio: float | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {'splits': {}, 'image_root_candidates': []}
    for split in ('train', 'valid', 'test'):
        split_data = load_split_dataframe(split, config, strict_missing_ratio=strict_missing_ratio)
        summary['splits'][split] = {
            'total_rows': split_data.total_rows,
            'available_rows': split_data.available_rows,
            'missing_rows': split_data.missing_rows,
            'label_distribution': split_data.label_distribution,
            'label_folder_mismatches': split_data.label_folder_mismatches,
            'resolved_image_root': str(split_data.image_root),
        }
    return summary


def create_split_sequence(
    split_data: SplitData,
    config: AppConfig,
    model_type: str,
    training: bool,
    shuffle: bool,
    batch_size: int | None = None,
    image_size: int | None = None,
    cache_images: bool | None = None,
    rgb_input_only: bool = False,
    auxiliary_quality: bool = False,
    class_rgb_stats: dict[int, dict[str, np.ndarray]] | None = None,
    equalize_ycrcb: bool | None = None,
    match_fake_to_real_stats: bool | None = None,
) -> CSVImageSequence:
    eq = config.equalize_ycrcb if equalize_ycrcb is None else equalize_ycrcb
    match_fr = config.match_fake_to_real_stats if match_fake_to_real_stats is None else match_fake_to_real_stats
    return CSVImageSequence(
        dataframe=split_data.dataframe,
        batch_size=batch_size or config.batch_size,
        image_size=image_size or config.image_size,
        model_type=model_type,
        training=training,
        shuffle=shuffle,
        seed=config.random_seed,
        aug_hflip_prob=config.aug_hflip_prob,
        aug_brightness_delta=config.aug_brightness_delta,
        aug_contrast_low=config.aug_contrast_low,
        aug_contrast_high=config.aug_contrast_high,
        cache_images=config.cache_images if cache_images is None else cache_images,
        max_cache_images=config.max_cache_images,
        use_face_crop=config.use_face_crop,
        face_crop_expand=config.face_crop_expand,
        rgb_input_only=rgb_input_only,
        auxiliary_quality=auxiliary_quality,
        class_rgb_stats=class_rgb_stats,
        equalize_ycrcb=eq,
        match_fake_to_real_stats=match_fr,
    )


def build_all_sequences(
    config: AppConfig,
    model_type: str,
    batch_size: int | None = None,
    image_size: int | None = None,
) -> tuple[CSVImageSequence, CSVImageSequence, CSVImageSequence, dict[str, SplitData]]:
    splits: dict[str, SplitData] = {
        'train': load_split_dataframe('train', config),
        'valid': load_split_dataframe('valid', config),
        'test': load_split_dataframe('test', config),
    }

    train_seq = create_split_sequence(
        split_data=splits['train'],
        config=config,
        model_type=model_type,
        training=True,
        shuffle=True,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=False,
    )
    valid_seq = create_split_sequence(
        split_data=splits['valid'],
        config=config,
        model_type=model_type,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=True,
    )
    test_seq = create_split_sequence(
        split_data=splits['test'],
        config=config,
        model_type=model_type,
        training=False,
        shuffle=False,
        batch_size=batch_size,
        image_size=image_size,
        cache_images=True,
    )

    return train_seq, valid_seq, test_seq, splits
