"""
Post-training sanity check: Wikipedia (expected REAL) vs TPDNE (expected FAKE) with confusion matrix.
"""
from __future__ import annotations

import json
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import get_config
from utils.inference_utils import load_image_from_path, predict_pil_image
from utils.model_utils import load_cnn_model, load_optional_joblib, load_xgb_model

USER_AGENT = 'Mozilla/5.0 (compatible; HybridDeepfakeDetector/1.1; +auto-val)'


def _download_wiki_thumb(title: str, dest: Path) -> bool:
    safe = urllib.parse.quote(title.replace(' ', '_'), safe='')
    api = (
        f'https://en.wikipedia.org/w/api.php?action=query&titles={safe}'
        f'&prop=pageimages&format=json&pithumbsize=800'
    )
    req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=35.0) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        pages = (payload.get('query') or {}).get('pages') or {}
        for _pid, page in pages.items():
            src = ((page or {}).get('thumbnail') or {}).get('source')
            if src:
                r2 = urllib.request.Request(src, headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(r2, timeout=45.0) as img:
                    dest.write_bytes(img.read())
                return dest.stat().st_size > 800
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError):
        pass
    return False


def _download_tpdne(dest: Path) -> bool:
    try:
        req = urllib.request.Request('https://thispersondoesnotexist.com/', headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            dest.write_bytes(resp.read())
        return dest.stat().st_size > 800
    except (OSError, urllib.error.URLError):
        return False


def main() -> None:
    cfg = get_config()
    tmp = Path(tempfile.mkdtemp(prefix='hdfd_autoval_'))

    y_true: list[int] = []
    y_pred: list[int] = []
    labels_txt: list[str] = []

    cnn = load_cnn_model(cfg.cnn_model_path)
    xgb = load_xgb_model(cfg.xgb_model_path)
    scaler = load_optional_joblib(cfg.xgb_scaler_path)

    celebs = [
        ('Shah Rukh Khan', 1),
        ('Albert Einstein', 1),
        ('Oprah Winfrey', 1),
    ]
    for name, lab in celebs:
        p = tmp / f'real_{name.replace(" ", "_")}.jpg'
        if _download_wiki_thumb(name, p):
            img = load_image_from_path(p)
            out = predict_pil_image(
                image=img,
                model_choice='hybrid',
                config=cfg,
                cnn_model=cnn,
                xgb_model=xgb,
                feature_scaler=scaler,
            )
            pr = float((out.get('probabilities') or {}).get('real', 0.0))
            hy_thr = float(out.get('decision_threshold', 0.6))
            pred = 1 if pr >= hy_thr else 0
            y_true.append(lab)
            y_pred.append(pred)
            labels_txt.append(f'wiki:{name}')

    for i in range(3):
        p = tmp / f'fake_{i}.jpg'
        if _download_tpdne(p):
            img = load_image_from_path(p)
            out = predict_pil_image(
                image=img,
                model_choice='hybrid',
                config=cfg,
                cnn_model=cnn,
                xgb_model=xgb,
                feature_scaler=scaler,
            )
            pr = float((out.get('probabilities') or {}).get('real', 0.0))
            hy_thr = float(out.get('decision_threshold', 0.6))
            pred = 1 if pr >= hy_thr else 0
            y_true.append(0)
            y_pred.append(pred)
            labels_txt.append(f'tpdne:{i}')

    if not y_true:
        print('automated_validation: no samples downloaded; skip matrix.')
        return

    yt = np.array(y_true)
    yp = np.array(y_pred)
    print('\n=== Automated validation (hybrid, threshold from metadata) ===\n')
    for i, t in enumerate(labels_txt):
        print(f'{t}: true={yt[i]} pred={yp[i]}')

    print('\nConfusion matrix [rows=true Fake/Real order 0,1]:')
    print(confusion_matrix(yt, yp, labels=[0, 1]))
    print('\nClassification report:')
    print(classification_report(yt, yp, labels=[0, 1], target_names=['fake', 'real'], zero_division=0))

    out_path = cfg.results_dir / 'automated_validation_report.txt'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['automated_validation', str(confusion_matrix(yt, yp, labels=[0, 1])), classification_report(yt, yp, labels=[0, 1], target_names=['fake', 'real'], zero_division=0)]
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
