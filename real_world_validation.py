"""
Fetch a few real-world images (Wikimedia / Wikipedia) and synthetic-style faces, run
the same preprocessing + hybrid inference, and print a summary table.

No cloud APIs: uses public HTTPS endpoints (Wikipedia/Wikimedia) and optional TPDNE.
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from utils.config import get_config
from utils.inference_utils import load_image_from_path, predict_pil_image
from utils.model_utils import load_cnn_model, load_optional_joblib, load_xgb_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('real_world_validation')

USER_AGENT = 'Mozilla/5.0 (compatible; HybridDeepfakeDetector/1.0; +validation)'


def _download(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=45.0) as resp:
            dest.write_bytes(resp.read())
        return dest.stat().st_size > 200
    except (urllib.error.URLError, OSError):
        return False


def _wikipedia_thumb(title: str, out: Path) -> bool:
    safe = urllib.parse.quote(title.replace(' ', '_'))
    api = (
        f'https://en.wikipedia.org/w/api.php?action=query&titles={safe}'
        f'&prop=pageimages&format=json&pithumbsize=640'
    )
    try:
        req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        pages = (payload.get('query') or {}).get('pages') or {}
        for _pid, page in pages.items():
            thumb = (page or {}).get('thumbnail') or {}
            src = thumb.get('source')
            if src:
                return _download(src, out)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        pass
    return False


def _commons_first_image(search: str, out: Path) -> bool:
    """Use Wikimedia Commons API search (bitmap files) — proxy for broad web image discovery."""
    params = urllib.parse.urlencode(
        {
            'action': 'query',
            'list': 'search',
            'srsearch': search,
            'format': 'json',
            'srnamespace': '6',
            'srlimit': '5',
        }
    )
    api = f'https://commons.wikimedia.org/w/api.php?{params}'
    try:
        req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        hits = ((payload.get('query') or {}).get('search')) or []
        for hit in hits:
            title = hit.get('title', '')
            if not title.startswith('File:'):
                continue
            file_name = title.replace('File:', '')
            info_params = urllib.parse.urlencode(
                {
                    'action': 'query',
                    'titles': f'File:{file_name}',
                    'prop': 'imageinfo',
                    'iiprop': 'url',
                    'format': 'json',
                }
            )
            info_url = f'https://commons.wikimedia.org/w/api.php?{info_params}'
            ireq = urllib.request.Request(info_url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(ireq, timeout=30.0) as ires:
                info = json.loads(ires.read().decode('utf-8'))
            pages = (info.get('query') or {}).get('pages') or {}
            for _p, pdata in pages.items():
                infos = (pdata or {}).get('imageinfo') or []
                if infos and infos[0].get('url'):
                    url = infos[0]['url']
                    lower = url.lower()
                    if any(lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')):
                        return _download(url, out)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        pass
    return False


def _tpdne(out: Path) -> bool:
    return _download('https://thispersondoesnotexist.com/', out)


def main() -> None:
    cfg = get_config()
    tmp = Path(tempfile.mkdtemp(prefix='hdfd_rw_'))

    samples: list[tuple[str, str, Path]] = []
    # Wikipedia thumbnails (expect REAL) — include celebrity portrait (e.g. SRK) for real-world check
    for title, tag in [
        ('Shah Rukh Khan', 'wiki_real'),
        ('Albert Einstein', 'wiki_real'),
        ('Oprah Winfrey', 'wiki_real'),
    ]:
        p = tmp / f'{tag}_{title.replace(" ", "_")}.jpg'
        if _wikipedia_thumb(title, p):
            samples.append((tag, f'Wikipedia:{title}', p))

    # Commons search for additional real portraits
    p = tmp / 'commons_portrait.jpg'
    if _commons_first_image('portrait photograph face', p):
        samples.append(('commons_real', 'Commons search: portrait photograph face', p))

    # Synthetic / manipulation proxies (expect FAKE or high uncertainty)
    p = tmp / 'tpdne.jpg'
    if _tpdne(p):
        samples.append(('tpdne_fake', 'thispersondoesnotexist.com (synthetic)', p))

    p = tmp / 'commons_deepfake.jpg'
    if _commons_first_image('deepfake face', p):
        samples.append(('commons_df', 'Commons search: deepfake face', p))

    if not samples:
        raise RuntimeError('Could not download any validation images (network blocked?).')

    cnn = load_cnn_model(cfg.cnn_model_path)
    xgb = load_xgb_model(cfg.xgb_model_path)
    scaler = load_optional_joblib(cfg.xgb_scaler_path)

    print('\n=== Real-world validation (hybrid) ===\n')
    print(f'{"Source":<40} {"Class":<12} {"Conf":<8} {"P(real)"}')
    print('-' * 78)

    for tag, label, path in samples:
        try:
            img = load_image_from_path(path)
            out = predict_pil_image(
                image=img,
                model_choice='hybrid',
                config=cfg,
                cnn_model=cnn,
                xgb_model=xgb,
                feature_scaler=scaler,
            )
            cls = str(out.get('predicted_class', ''))
            conf = float(out.get('confidence', 0.0))
            preal = float((out.get('probabilities') or {}).get('real', 0.0))
            print(f'{label[:40]:<40} {cls:<12} {conf:.4f}   {preal:.4f}')
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('Failed on %s: %s', path, exc)
            print(f'{label[:40]:<40} {"ERROR":<12} {"-":<8} {"-"}')

    print('\nNote: Labels are qualitative; synthetic faces should skew toward Fake if training used similar proxies.\n')


if __name__ == '__main__':
    main()
