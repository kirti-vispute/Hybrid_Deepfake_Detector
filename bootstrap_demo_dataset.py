"""
Automated balanced dataset under dataset/real and dataset/fake + CSV splits in data/.

Sources (REAL): RandomUser API, Wikipedia thumbnails (celebrities), Wikimedia Commons search.
Sources (FAKE): thispersondoesnotexist.com, augmented proxies if needed.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from utils.face_detection import image_has_detectable_face
from utils.web_image_search import ddg_image_urls

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s - %(message)s')
LOGGER = logging.getLogger('bootstrap_demo_dataset')

USER_AGENT = 'Mozilla/5.0 (compatible; HybridDeepfakeDetector/1.1; +auto-dataset)'

WIKI_SEARCH_QUERIES = ['actors', 'politicians', 'scientists', 'people']

DDG_FACE_QUERIES = [
    'human face portrait',
    'celebrity face',
    'person face close up',
]

WIKI_CELEBRITY_TITLES = [
    'Shah Rukh Khan',
    'Virat Kohli',
    'Albert Einstein',
    'Oprah Winfrey',
    'Leonardo DiCaprio',
    'Taylor Swift',
    'Barack Obama',
    'Queen Elizabeth II',
    'Cristiano Ronaldo',
    'Serena Williams',
    'Elon Musk',
    'Malala Yousafzai',
    'Priyanka Chopra',
    'Tom Hanks',
    'Angela Merkel',
]


def _download(url: str, dest: Path, timeout: float = 45.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 512:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        LOGGER.warning('Download failed %s -> %s (%s)', url, dest, exc)
        return False


def _fetch_randomuser_batch(n: int) -> list[str]:
    urls: list[str] = []
    remaining = n
    page = 1
    while remaining > 0:
        batch = min(500, remaining)
        api = f'https://randomuser.me/api/?results={batch}&page={page}'
        try:
            req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            LOGGER.error('randomuser API failed: %s', exc)
            break
        for r in payload.get('results', []):
            pic = (r.get('picture') or {}).get('large')
            if pic:
                urls.append(pic)
        remaining = n - len(urls)
        page += 1
        time.sleep(0.12)
    return urls[:n]


def _wikipedia_search_titles(query: str, limit: int = 15) -> list[str]:
    params = urllib.parse.urlencode(
        {'action': 'query', 'list': 'search', 'srsearch': query, 'format': 'json', 'srlimit': str(limit)}
    )
    api = f'https://en.wikipedia.org/w/api.php?{params}'
    try:
        req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=35.0) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        hits = ((payload.get('query') or {}).get('search')) or []
        return [str(h.get('title', '')) for h in hits if h.get('title')]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return []


def _wikipedia_thumb_to_file(title: str, dest: Path) -> bool:
    safe = urllib.parse.quote(title.replace(' ', '_'), safe='')
    api = (
        f'https://en.wikipedia.org/w/api.php?action=query&titles={safe}'
        f'&prop=pageimages&format=json&pithumbsize=800'
    )
    try:
        req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=35.0) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
        pages = (payload.get('query') or {}).get('pages') or {}
        for _pid, page in pages.items():
            thumb = (page or {}).get('thumbnail') or {}
            src = thumb.get('source')
            if src:
                return _download(src, dest)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        pass
    return False


def _download_if_face(dest: Path, url: str) -> bool:
    if not _download(url, dest):
        return False
    try:
        with Image.open(dest) as im:
            rgb = im.convert('RGB')
        if not image_has_detectable_face(rgb):
            dest.unlink(missing_ok=True)
            return False
        return True
    except OSError:
        dest.unlink(missing_ok=True)
        return False


def _download_ddg_face_portraits(dest_dir: Path, target: int, start_idx: int) -> list[str]:
    rels: list[str] = []
    urls = ddg_image_urls(DDG_FACE_QUERIES, max_per_query=max(80, target // 3), total_cap=max(target * 3, 600))
    i = start_idx
    for url in urls:
        if len(rels) >= target:
            break
        dest = dest_dir / f'ddg_{i:05d}.jpg'
        i += 1
        if _download_if_face(dest, url):
            rels.append(f'real/{dest.name}')
        time.sleep(0.05)
    return rels


def _download_wikipedia_search_faces(dest_dir: Path, target: int, start_idx: int) -> list[str]:
    rels: list[str] = []
    titles: list[str] = []
    for q in WIKI_SEARCH_QUERIES:
        titles.extend(_wikipedia_search_titles(q, limit=12))
        if len(titles) >= target * 2:
            break
    for j, title in enumerate(titles):
        if len(rels) >= target:
            break
        dest = dest_dir / f'ws_{start_idx + j:05d}.jpg'
        if _wikipedia_thumb_to_file(title, dest):
            try:
                with Image.open(dest) as im:
                    if not image_has_detectable_face(im.convert('RGB')):
                        dest.unlink(missing_ok=True)
                        continue
            except OSError:
                dest.unlink(missing_ok=True)
                continue
            rels.append(f'real/{dest.name}')
        time.sleep(0.1)
    return rels


def _commons_first_portrait(dest: Path, offset: int) -> bool:
    params = urllib.parse.urlencode(
        {
            'action': 'query',
            'list': 'search',
            'srsearch': 'portrait photograph face',
            'format': 'json',
            'srnamespace': '6',
            'srlimit': '10',
            'sroffset': str(offset % 40),
        }
    )
    api = f'https://commons.wikimedia.org/w/api.php?{params}'
    try:
        req = urllib.request.Request(api, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(req, timeout=35.0) as resp:
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
            with urllib.request.urlopen(ireq, timeout=35.0) as ires:
                info = json.loads(ires.read().decode('utf-8'))
            pages = (info.get('query') or {}).get('pages') or {}
            for _p, pdata in pages.items():
                infos = (pdata or {}).get('imageinfo') or []
                if infos and infos[0].get('url'):
                    url = infos[0]['url']
                    lower = url.lower()
                    if any(lower.endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')):
                        return _download(url, dest)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        pass
    return False


def _make_proxy_fake_from_real(src: Path, dst: Path) -> bool:
    try:
        with Image.open(src) as im:
            img = im.convert('RGB').resize((256, 256), Image.BILINEAR)
            img = ImageEnhance.Color(img).enhance(1.45)
            img = ImageEnhance.Contrast(img).enhance(1.25)
            img = img.filter(ImageFilter.GaussianBlur(radius=2.2))
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, quality=72)
        return True
    except OSError:
        return False


def _download_tpdne_faces(dest_dir: Path, count: int, delay_s: float = 0.3) -> list[str]:
    rel_paths: list[str] = []
    url = 'https://thispersondoesnotexist.com/'
    for i in range(count):
        out = dest_dir / f'f_{i:05d}.jpg'
        if _download(url, out):
            rel_paths.append(f'fake/{out.name}')
        time.sleep(delay_s)
    return rel_paths


def _download_real_faces(dest_dir: Path, count: int) -> list[str]:
    rel_paths: list[str] = []
    urls = _fetch_randomuser_batch(count)
    for i, u in enumerate(urls):
        out = dest_dir / f'r_{i:05d}.jpg'
        if _download(u, out):
            rel_paths.append(f'real/{out.name}')
        time.sleep(0.04)
    return rel_paths


def _download_wikipedia_rotating(dest_dir: Path, count: int, start_idx: int) -> list[str]:
    rels: list[str] = []
    titles = WIKI_CELEBRITY_TITLES
    for i in range(count):
        title = titles[i % len(titles)]
        dest = dest_dir / f'wiki_{start_idx + i:05d}.jpg'
        if _wikipedia_thumb_to_file(title, dest):
            rels.append(f'real/{dest.name}')
        time.sleep(0.08)
    return rels


def _fill_fake_with_proxies(real_dir: Path, fake_dir: Path, need: int, offset: int) -> list[str]:
    rels: list[str] = []
    real_files = sorted(real_dir.glob('*.jpg'))
    if not real_files or need <= 0:
        return rels
    for i in range(need):
        src = real_files[i % len(real_files)]
        dst = fake_dir / f'proxy_{offset + i:05d}.jpg'
        if _make_proxy_fake_from_real(src, dst):
            rels.append(f'fake/{dst.name}')
    return rels


def _write_splits(rows: list[dict], seed: int = 42) -> None:
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)
    train = df.iloc[:n_train]
    valid = df.iloc[n_train : n_train + n_valid]
    test = df.iloc[n_train + n_valid :]

    cfg_root = PROJECT_ROOT / 'data'
    cfg_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(cfg_root / 'train.csv', index=False)
    valid.to_csv(cfg_root / 'valid.csv', index=False)
    test.to_csv(cfg_root / 'test.csv', index=False)
    LOGGER.info('Wrote train=%d valid=%d test=%d rows under %s', len(train), len(valid), len(test), cfg_root)


def main(per_class: int = 520) -> None:
    image_root = PROJECT_ROOT / 'dataset'
    real_dir = image_root / 'real'
    fake_dir = image_root / 'fake'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    n_rand = max(0, min(per_class // 4, 200))
    n_ddg = max(0, min(520, max(per_class // 2, per_class - n_rand - 80)))
    n_wiki = max(0, per_class - n_rand - n_ddg)
    n_commons = max(0, min(120, per_class // 4))

    LOGGER.info(
        'Building ~%d real (randomuser=%d ddg~=%d wiki~=%d commons~=%d) + ~%d fake ...',
        per_class,
        n_rand,
        n_ddg,
        n_wiki,
        n_commons,
        per_class,
    )

    real_rels: list[str] = []
    real_rels.extend(_download_real_faces(real_dir, n_rand))
    real_rels.extend(_download_ddg_face_portraits(real_dir, n_ddg, start_idx=len(real_rels)))
    real_rels.extend(_download_wikipedia_rotating(real_dir, n_wiki, start_idx=len(real_rels)))
    real_rels.extend(_download_wikipedia_search_faces(real_dir, max(0, min(80, per_class // 6)), start_idx=len(real_rels)))

    for c in range(n_commons):
        dest = real_dir / f'commons_{c:05d}.jpg'
        if _commons_first_portrait(dest, offset=c * 3):
            real_rels.append(f'real/{dest.name}')
        time.sleep(0.1)

    fill_i = 0
    while len(real_rels) < per_class and fill_i < per_class * 2:
        i = len(real_rels)
        dest = real_dir / f'fill_{fill_i:05d}.jpg'
        ok = False
        if _commons_first_portrait(dest, offset=fill_i * 7):
            ok = True
        elif _wikipedia_thumb_to_file(WIKI_CELEBRITY_TITLES[fill_i % len(WIKI_CELEBRITY_TITLES)], dest):
            ok = True
        if ok:
            real_rels.append(f'real/{dest.name}')
        fill_i += 1
        time.sleep(0.08)

    real_rels = real_rels[:per_class]

    fake_rels = _download_tpdne_faces(fake_dir, per_class)
    if len(fake_rels) < per_class // 2:
        LOGGER.warning('TPDNE short (%d); adding proxy-fakes.', len(fake_rels))
        fake_rels.extend(_fill_fake_with_proxies(real_dir, fake_dir, need=per_class - len(fake_rels), offset=len(fake_rels)))

    fake_rels = fake_rels[:per_class]
    n_bal = min(len(real_rels), len(fake_rels))
    real_rels = real_rels[:n_bal]
    fake_rels = fake_rels[:n_bal]

    rows: list[dict] = []
    for rel in real_rels:
        rows.append({'path': rel.replace('\\', '/'), 'label': 1, 'label_str': 'real'})
    for rel in fake_rels:
        rows.append({'path': rel.replace('\\', '/'), 'label': 0, 'label_str': 'fake'})

    if len(rows) < 200:
        raise RuntimeError(
            f'Too few images collected ({len(rows)}). Check network access or try again later.'
        )

    _write_splits(rows)
    LOGGER.info('Bootstrap complete. Total usable rows: %d (target per class=%d)', len(rows) // 2, per_class)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Create train/valid/test CSVs and download images into dataset/.')
    p.add_argument('--per-class', type=int, default=520, help='Target images per class (real vs fake).')
    args = p.parse_args()
    main(per_class=args.per_class)
