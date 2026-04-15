"""
Automated image URL discovery (no API keys). Uses DuckDuckGo image search as a practical
stand-in for scripted Google Images access.
"""
from __future__ import annotations

import logging
from typing import Iterable

LOGGER = logging.getLogger(__name__)


def ddg_image_urls(queries: Iterable[str], max_per_query: int, total_cap: int) -> list[str]:
    """Return unique image URLs from DuckDuckGo image search."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        LOGGER.warning('duckduckgo-search not installed; skipping DDG image discovery.')
        return []

    urls: list[str] = []
    seen: set[str] = set()
    for q in queries:
        if len(urls) >= total_cap:
            break
        try:
            with DDGS() as ddgs:
                results = ddgs.images(str(q), max_results=max_per_query)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning('DDG images query failed (%s): %s', q, exc)
            continue
        for r in results or []:
            u = (r or {}).get('image') or (r or {}).get('thumbnail')
            if not u or u in seen:
                continue
            seen.add(u)
            urls.append(str(u))
            if len(urls) >= total_cap:
                break
    return urls
