"""
Face detection (OpenCV Haar) with center-crop fallback for deepfake inference/training.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)

_CASCADE = None


def _cascade():
    global _CASCADE  # pylint: disable=global-statement
    if _CASCADE is None:
        try:
            import cv2
        except ImportError:
            LOGGER.warning('OpenCV not installed; face detection disabled (install opencv-python-headless).')
            _CASCADE = False
            return _CASCADE
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _CASCADE = cv2.CascadeClassifier(path)
        if _CASCADE.empty():
            LOGGER.warning('Haar cascade failed to load from %s', path)
            _CASCADE = False
    return _CASCADE


def _largest_face_box(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    try:
        import cv2
    except ImportError:
        return None

    cascade = _cascade()
    if cascade is None or cascade is False or cascade.empty():
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda r: int(r[2]) * int(r[3]))


def _center_square_crop(w: int, h: int, frac: float = 0.85) -> tuple[int, int, int, int]:
    side = int(min(w, h) * frac)
    cx, cy = w // 2, h // 2
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    if x2 - x1 < side:
        x1 = max(0, x2 - side)
    if y2 - y1 < side:
        y1 = max(0, y2 - side)
    return x1, y1, x2, y2


def _expand_square_box(
    x: int,
    y: int,
    fw: int,
    fh: int,
    w: int,
    h: int,
    expand: float,
) -> tuple[int, int, int, int]:
    cx = x + fw / 2.0
    cy = y + fh / 2.0
    side = max(fw, fh) * float(expand)
    half = side / 2.0
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(w, cx + half))
    y2 = int(min(h, cy + half))
    return x1, y1, x2, y2


def image_has_detectable_face(pil_image: Image.Image) -> bool:
    """True if Haar finds a frontal face; True when OpenCV unavailable (do not drop downloads)."""
    try:
        import cv2
    except ImportError:
        return True
    rgb = pil_image.convert('RGB')
    arr = np.asarray(rgb)
    try:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except Exception:  # pylint: disable=broad-except
        return True
    return _largest_face_box(gray) is not None


def extract_face_or_fallback(pil_image: Image.Image, expand: float = 1.25) -> tuple[Image.Image, dict[str, Any]]:
    """
    Return largest-face crop (RGB) or center crop if no face. Meta describes the decision.
    """
    rgb = pil_image.convert('RGB')
    arr = np.asarray(rgb)
    h, w = arr.shape[0], arr.shape[1]

    try:
        import cv2

        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    except ImportError:
        x1, y1, x2, y2 = _center_square_crop(w, h)
        return rgb.crop((x1, y1, x2, y2)), {'face_detected': False, 'method': 'center_crop', 'reason': 'no_opencv'}
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning('OpenCV convert failed (%s); using center crop.', exc)
        x1, y1, x2, y2 = _center_square_crop(w, h)
        return rgb.crop((x1, y1, x2, y2)), {'face_detected': False, 'method': 'center_crop', 'reason': 'cv_error'}

    box = _largest_face_box(gray)
    if box is None:
        x1, y1, x2, y2 = _center_square_crop(w, h)
        return rgb.crop((x1, y1, x2, y2)), {'face_detected': False, 'method': 'center_crop', 'reason': 'no_face'}

    x, y, fw, fh = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1, y1, x2, y2 = _expand_square_box(x, y, fw, fh, w, h, expand)
    cropped = rgb.crop((x1, y1, x2, y2))
    return cropped, {
        'face_detected': True,
        'method': 'haar_face',
        'box': [x, y, fw, fh],
        'crop': [x1, y1, x2, y2],
    }


def resize_rgb(pil_image: Image.Image, size: int) -> Image.Image:
    return pil_image.convert('RGB').resize((size, size), Image.BILINEAR)
