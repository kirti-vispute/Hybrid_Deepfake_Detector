"""
2D visualization of CNN embeddings after StandardScaler + PCA (same as hybrid pipeline).
Saves PNGs under results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from utils.config import get_config
from utils.model_utils import load_optional_joblib, load_optional_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='t-SNE / PCA plots for real vs fake embeddings.')
    p.add_argument('--split', choices=['train', 'valid', 'test'], default='valid')
    p.add_argument('--max-samples', type=int, default=800)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    meta = load_optional_json(cfg.hybrid_metadata_path) or {}
    pca_path = meta.get('pca_path', str(cfg.hybrid_pca_path))
    scaler = load_optional_joblib(cfg.xgb_scaler_path)
    pca = load_optional_joblib(Path(pca_path))

    npz_path = cfg.feature_dir / f'{args.split}_features.npz'
    if not npz_path.exists():
        print(f'Missing {npz_path}; run feature extraction first.', file=sys.stderr)
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    X, y = data['features'], data['labels'].astype(int)

    if scaler is not None:
        X = scaler.transform(X)
    if pca is not None:
        Xp = pca.transform(X)
    else:
        Xp = X

    n = min(args.max_samples, len(y))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(y), size=n, replace=False)
    Xs = Xp[idx]
    ys = y[idx]

    if Xs.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        real_m = ys == 1
        fake_m = ys == 0
        ax.scatter(Xs[real_m, 0], Xs[real_m, 1], s=8, alpha=0.55, c='#1f77b4', label='real')
        ax.scatter(Xs[fake_m, 0], Xs[fake_m, 1], s=8, alpha=0.55, c='#d62728', label='fake')
        ax.set_title('PCA space (first two components)')
        ax.legend()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        out_pca = cfg.results_dir / 'embedding_pca2d.png'
        fig.tight_layout()
        fig.savefig(out_pca, dpi=140)
        plt.close(fig)
        print('Wrote', out_pca)

    perp = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=float(perp),
        init='pca',
        learning_rate='auto',
    )
    Z = tsne.fit_transform(Xs.astype(np.float64))

    fig, ax = plt.subplots(figsize=(7, 6))
    real_m = ys == 1
    fake_m = ys == 0
    ax.scatter(Z[real_m, 0], Z[real_m, 1], s=8, alpha=0.55, c='#1f77b4', label='real')
    ax.scatter(Z[fake_m, 0], Z[fake_m, 1], s=8, alpha=0.55, c='#d62728', label='fake')
    ax.set_title(f't-SNE of hybrid PCA features ({args.split}, n={n})')
    ax.legend()
    out_tsne = cfg.results_dir / 'embedding_tsne.png'
    fig.tight_layout()
    fig.savefig(out_tsne, dpi=140)
    plt.close(fig)
    print('Wrote', out_tsne)


if __name__ == '__main__':
    main()
