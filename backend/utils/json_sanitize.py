"""Ensure prediction payloads are JSON-serializable (no numpy scalar leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np


def sanitize_for_json(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
