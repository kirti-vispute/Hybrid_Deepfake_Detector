"""
One-command entry point: bootstrap demo data (optional), train pipeline, evaluate, real-world check.

Examples:
  python run_project.py --all
  python run_project.py --bootstrap-only --per-class 400
  python run_project.py --train-only --mode fast --backbone efficientnetb0
  python run_project.py --backend-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _py() -> str:
    return sys.executable


def _run(cmd: list[str], env: dict | None = None) -> None:
    printable = ' '.join(cmd)
    print(f'\n[RUN] {printable}\n', flush=True)
    merged = os.environ.copy()
    keras_home = PROJECT_ROOT / '.keras'
    keras_home.mkdir(parents=True, exist_ok=True)
    merged.setdefault('KERAS_HOME', str(keras_home))
    merged.setdefault('TFHUB_CACHE_DIR', str(keras_home / 'tfhub'))
    merged.setdefault('KERAS_BACKEND', 'torch')
    if env:
        merged.update(env)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=merged)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Hybrid Deepfake Detector — full local pipeline')
    p.add_argument('--all', action='store_true', help='Bootstrap (if needed), train, evaluate, real-world validation.')
    p.add_argument('--bootstrap-only', action='store_true')
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--real-world-only', action='store_true')
    p.add_argument('--backend-only', action='store_true', help='Start Flask backend only (models must exist).')

    p.add_argument('--per-class', type=int, default=520, help='Images per class for bootstrap demo dataset.')
    p.add_argument('--mode', choices=['debug', 'fast', 'strong', 'rigorous'], default='fast')
    p.add_argument(
        '--backbone',
        choices=['efficientnetb0', 'efficientnetb1', 'resnet50', 'mobilenetv2', 'vgg16'],
        default='efficientnetb0',
    )
    p.add_argument('--force-bootstrap', action='store_true', help='Re-download demo images even if data/train.csv exists.')
    return p.parse_args()


def _data_ready() -> bool:
    train_csv = PROJECT_ROOT / 'data' / 'train.csv'
    return train_csv.exists() and train_csv.stat().st_size > 50


def main() -> None:
    args = parse_args()
    py = _py()

    if args.backend_only:
        _run([py, str(PROJECT_ROOT / 'backend' / 'app.py')])
        return

    do_bootstrap = args.all or args.bootstrap_only
    do_train = args.all or args.train_only
    do_eval = args.all or args.evaluate_only
    do_rw = args.all or args.real_world_only

    if args.all:
        do_eval = True
        do_rw = True
        do_train = True
        do_bootstrap = True

    if do_bootstrap:
        if args.force_bootstrap or not _data_ready():
            _run([py, str(PROJECT_ROOT / 'bootstrap_demo_dataset.py'), '--per-class', str(args.per_class)])

    if do_train:
        _run(
            [
                py,
                str(PROJECT_ROOT / 'run_training.py'),
                '--mode',
                args.mode,
                '--backbone-name',
                args.backbone,
            ]
        )

    if do_eval and not do_train:
        _run([py, str(PROJECT_ROOT / 'evaluate.py')])
        _run([py, str(PROJECT_ROOT / 'visualize_embeddings.py'), '--split', 'valid'])

    if do_rw:
        _run([py, str(PROJECT_ROOT / 'real_world_validation.py')])
        _run([py, str(PROJECT_ROOT / 'automated_validation.py')])


if __name__ == '__main__':
    main()
