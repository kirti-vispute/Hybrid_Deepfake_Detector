from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable


def normalize_csv_relative_path(relative_path: str) -> Path:
    clean = str(relative_path).strip().replace('\\', '/')
    parts = PurePosixPath(clean).parts
    return Path(*parts)


def build_absolute_image_path(image_root: Path, relative_path: str) -> Path:
    return image_root / normalize_csv_relative_path(relative_path)


def _report_roots_from_json(report_path: Path) -> list[Path]:
    if not report_path.exists():
        return []
    try:
        payload = json.loads(report_path.read_text(encoding='utf-8-sig'))
    except (OSError, json.JSONDecodeError):
        return []

    roots: list[Path] = []
    splits = payload.get('splits')
    if isinstance(splits, dict):
        for split_payload in splits.values():
            if not isinstance(split_payload, dict):
                continue
            root_str = split_payload.get('resolved_image_root')
            if root_str:
                roots.append(Path(str(root_str)))
    image_root = payload.get('image_root')
    if image_root:
        roots.append(Path(str(image_root)))
    return roots


def _known_dataset_roots(project_root: Path) -> list[Path]:
    results_dir = project_root / 'results'
    candidates: list[Path] = []
    for report_name in ('dataset_audit_kaggle140k.json', 'dataset_audit.json', 'kaggle_dataset_summary.json'):
        candidates.extend(_report_roots_from_json(results_dir / report_name))
    candidates.extend(
        [
            project_root / 'Dataset',
            project_root / 'Dataset' / 'rvf10k',
            project_root / 'dataset',
            project_root / 'dataset' / 'real-vs-fake',
        ]
    )
    return candidates


def resolve_candidate_image_roots(image_root: Path) -> list[Path]:
    project_root = Path(__file__).resolve().parents[1]
    candidates = [image_root, image_root / 'real-vs-fake', *_known_dataset_roots(project_root)]
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
            if build_absolute_image_path(candidate, rel).exists():
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_root = candidate

    return best_root


def resolve_absolute_image_path(image_root: Path, relative_path: str) -> Path:
    relative = normalize_csv_relative_path(relative_path)
    for candidate in resolve_candidate_image_roots(image_root):
        full_path = candidate / relative
        if full_path.exists():
            return full_path
    return build_absolute_image_path(resolve_existing_image_root(image_root, [relative_path]), relative_path)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        ensure_dir(path)
