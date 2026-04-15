from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from pathlib import Path

from utils.config import get_config
from utils.data_loader import audit_dataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('verify_dataset')


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description='Verify CSV files and referenced images before training.')
    parser.add_argument('--check-corrupt', action='store_true', help='Open images to detect corrupt files.')
    parser.add_argument(
        '--max-corrupt-checks-per-split',
        type=int,
        default=None,
        help='Optional limit for corrupt-image checks per split.',
    )
    parser.add_argument('--output-json', type=str, default=str(cfg.dataset_audit_path))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()

    output_json = Path(args.output_json).resolve()

    summary = audit_dataset(
        config=cfg,
        check_corrupt=args.check_corrupt,
        max_corrupt_checks_per_split=args.max_corrupt_checks_per_split,
        output_json=output_json,
    )

    print(json.dumps(summary, indent=2))
    LOGGER.info('Saved dataset audit report to %s', output_json)


if __name__ == '__main__':
    main()

