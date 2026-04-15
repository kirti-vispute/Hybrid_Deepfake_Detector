# Deploy: Vercel (frontend) + Render (backend)

## Layout

| Part | Path |
|------|------|
| Frontend | `frontend/` (Vite + React) |
| Backend | Run from **repo root**: `backend/`, `utils/`, `wsgi.py`, `models/` |

Do **not** set Render root directory to `backend/` only — `utils` and model paths resolve from the repository root.

## Render (Flask + ML)

1. **New Web Service** → connect this repo.
2. **Root directory:** *(empty — repo root)*.
3. **Build:** `pip install --upgrade pip && pip install -r requirements.txt`
4. **Start:** `gunicorn --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:$PORT wsgi:app`
5. **Health check path:** `/health`

### Render environment variables

| Key | Example |
|-----|---------|
| `CORS_ORIGINS` | `https://your-app.vercel.app` or comma-separated list; use `*` only for testing |
| `HDFD_*` | Optional — see `utils/config.py` for model paths |

### Model artifacts

Files under `models/` may be gitignored. Ship weights to the server (commit to private repo, upload via shell, persistent disk, or `HDFD_CNN_MODEL_PATH` / `HDFD_XGB_MODEL_PATH`).

TensorFlow + models need **enough RAM** (often ≥ 2 GB). First deploy without weights: `/health` works; `/api/predict` returns errors until artifacts exist.

## Vercel (frontend)

1. Import repo → **Root Directory:** `frontend`
2. **Environment variable:**

| Key | Value |
|-----|--------|
| `VITE_API_URL` | `https://<render-service>.onrender.com/api` |

3. Redeploy after changing env vars.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness |
| GET | `/api/health` | Liveness + model availability |
| GET | `/api/models` | UI availability |
| POST | `/api/predict` | Multipart `file`, form `model=hybrid` |

## Local

```bash
# Repo root
pip install -r requirements.txt
python backend/app.py
# or
gunicorn --workers 1 --threads 2 --timeout 120 --bind 127.0.0.1:8000 wsgi:app
```

```bash
cd frontend
npm install
# Optional: cp .env.example .env.local && edit VITE_API_URL
npm run dev
```

## Optional

Connect Render to this repo’s `render.yaml` for blueprint-based setup.
