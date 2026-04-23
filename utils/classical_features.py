from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from utils.face_detection import extract_face_or_fallback, resize_rgb


@dataclass(frozen=True)
class ClassicalFeatureSpec:
    color_hist_bins: int = 16
    gray_hist_bins: int = 32
    lbp_bins: int = 32
    resize_side: int = 128


@dataclass(frozen=True)
class IdentityFeatureSpec:
    face_size: int = 160
    lowres_side: int = 16
    classical: ClassicalFeatureSpec = ClassicalFeatureSpec(resize_side=128)


def _normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32).ravel()
    denom = float(np.sum(hist)) + 1e-8
    return hist / denom


def _lbp_histogram(gray: np.ndarray, bins: int) -> np.ndarray:
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    lbp = np.zeros_like(center, dtype=np.uint8)
    for idx, neigh in enumerate(neighbors):
        lbp |= ((neigh >= center).astype(np.uint8) << idx)
    hist, _ = np.histogram(lbp, bins=bins, range=(0, 256))
    return _normalize_hist(hist)


def _frequency_bands(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = float(np.max(r)) + 1e-8
    low = mag[r <= 0.20 * rmax].mean()
    mid = mag[(r > 0.20 * rmax) & (r <= 0.55 * rmax)].mean()
    high = mag[r > 0.55 * rmax].mean()
    return np.array([low, mid, high], dtype=np.float32)


def _channel_moments(bgr: np.ndarray) -> np.ndarray:
    feats = []
    for ch in cv2.split(bgr):
        x = ch.astype(np.float32)
        mean = float(np.mean(x))
        std = float(np.std(x))
        centered = x - mean
        skew = float(np.mean(np.sign(centered) * np.power(np.abs(centered), 3.0)) / (std**3 + 1e-8))
        feats.extend([mean, std, skew])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    for ch in cv2.split(hsv):
        x = ch.astype(np.float32)
        feats.extend([float(np.mean(x)), float(np.std(x))])
    return np.asarray(feats, dtype=np.float32)


def _gradient_hist(gray: np.ndarray, bins: int = 16) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.arctan2(gy, gx) + np.pi) * (bins / (2 * np.pi))
    idx = np.clip(ang.astype(np.int32), 0, bins - 1)
    hist = np.zeros((bins,), dtype=np.float32)
    for b in range(bins):
        hist[b] = float(np.sum(mag[idx == b]))
    return _normalize_hist(hist)


def _artifact_stats(bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))
    edge_mean = float(np.mean(edges))
    edge_std = float(np.std(edges))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_std = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))

    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        jpeg_mse = float(np.mean((bgr.astype(np.float32) - dec.astype(np.float32)) ** 2))
    else:
        jpeg_mse = 0.0

    return np.array(
        [edge_density, edge_mean, edge_std, lap_var, noise_std, jpeg_mse],
        dtype=np.float32,
    )


def extract_classical_features_from_bgr(
    bgr: np.ndarray,
    spec: ClassicalFeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or ClassicalFeatureSpec()
    bgr = cv2.resize(bgr, (spec.resize_side, spec.resize_side), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    color_hist = []
    for channel in cv2.split(bgr):
        hist, _ = np.histogram(channel, bins=spec.color_hist_bins, range=(0, 256))
        color_hist.append(_normalize_hist(hist))
    color_hist_vec = np.concatenate(color_hist).astype(np.float32)

    gray_hist, _ = np.histogram(gray, bins=spec.gray_hist_bins, range=(0, 256))
    gray_hist_vec = _normalize_hist(gray_hist)

    lbp_vec = _lbp_histogram(gray, bins=spec.lbp_bins)
    freq_vec = _frequency_bands(gray)
    moment_vec = _channel_moments(bgr)
    grad_vec = _gradient_hist(gray, bins=16)
    artifact_vec = _artifact_stats(bgr, gray)

    return np.concatenate(
        [color_hist_vec, gray_hist_vec, lbp_vec, freq_vec, moment_vec, grad_vec, artifact_vec]
    ).astype(np.float32)


def extract_classical_features_from_path(
    image_path: Path,
    spec: ClassicalFeatureSpec | None = None,
) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return extract_classical_features_from_bgr(bgr, spec=spec)


def _symmetry_features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    half = w // 2
    left = gray[:, :half]
    right = np.fliplr(gray[:, w - half :])
    min_h = min(left.shape[0], right.shape[0])
    min_w = min(left.shape[1], right.shape[1])
    left = left[:min_h, :min_w].astype(np.float32)
    right = right[:min_h, :min_w].astype(np.float32)
    diff = np.abs(left - right)
    return np.array([float(np.mean(diff)), float(np.std(diff)), float(np.percentile(diff, 90))], dtype=np.float32)


def _lowres_faceprint(gray: np.ndarray, lowres_side: int) -> np.ndarray:
    small = cv2.resize(gray, (lowres_side, lowres_side), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = (small - float(np.mean(small))) / (float(np.std(small)) + 1e-6)
    return small.reshape(-1).astype(np.float32)


def extract_identity_features_from_pil(
    image: Image.Image,
    spec: IdentityFeatureSpec | None = None,
    face_crop_expand: float = 1.25,
) -> np.ndarray:
    spec = spec or IdentityFeatureSpec()
    face, _meta = extract_face_or_fallback(image.convert('RGB'), expand=float(face_crop_expand))
    face = resize_rgb(face, spec.face_size)
    rgb = np.asarray(face, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    classical = extract_classical_features_from_bgr(bgr, spec=spec.classical)
    symmetry = _symmetry_features(gray)
    faceprint = _lowres_faceprint(gray, spec.lowres_side)
    return np.concatenate([classical, symmetry, faceprint]).astype(np.float32)


def extract_identity_features_from_path(
    image_path: Path,
    spec: IdentityFeatureSpec | None = None,
    face_crop_expand: float = 1.25,
) -> np.ndarray:
    spec = spec or IdentityFeatureSpec()
    try:
        with Image.open(image_path) as img:
            return extract_identity_features_from_pil(img.convert('RGB'), spec=spec, face_crop_expand=face_crop_expand)
    except (UnidentifiedImageError, OSError, ValueError):
        size = spec.classical.color_hist_bins * 3 + spec.classical.gray_hist_bins + spec.classical.lbp_bins + 3 + 15 + 16 + 6 + 3 + (spec.lowres_side * spec.lowres_side)
        return np.zeros((size,), dtype=np.float32)


def extract_identity_features_for_paths(
    paths: list[str] | np.ndarray,
    spec: IdentityFeatureSpec | None = None,
    face_crop_expand: float = 1.25,
) -> np.ndarray:
    spec = spec or IdentityFeatureSpec()
    features = [
        extract_identity_features_from_path(Path(str(path)), spec=spec, face_crop_expand=face_crop_expand)
        for path in paths
    ]
    return np.stack(features).astype(np.float32)
