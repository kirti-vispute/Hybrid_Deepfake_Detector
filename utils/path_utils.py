from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterable


def normalize_csv_relative_path(relative_path: str) -> Path:
    clean = str(relative_path).strip().replace('\\', '/')
    parts = PurePosixPath(clean).parts
    return Path(*parts)


def build_absolute_image_path(image_root: Path, relative_path: str) -> Path:
    # Avoid expensive per-path realpath resolution on large datasets.
    return image_root / normalize_csv_relative_path(relative_path)


def resolve_candidate_image_roots(image_root: Path) -> list[Path]:
    candidates = [image_root, image_root / 'real-vs-fake']
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def resolve_existing_image_root(image_root: Path, relative_paths: Iterable[str], sample_size: int = 500) -> Path:
    paths = []
    for idx, rel in enumerate(relative_paths):
        if idx >= sample_size:
            break
        paths.append(rel)

    if not paths:
        return image_root.resolve()

    best_root = image_root.resolve()
    best_hits = -1

    for candidate in resolve_candidate_image_roots(image_root):
        hits = 0
        for rel in paths:
            full_path = build_absolute_image_path(candidate, rel)
            if full_path.exists():
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_root = candidate

    return best_root


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)
