# Hybrid Deepfake Detector
 
An end-to-end image-based deepfake detection system using a CNN baseline and a hybrid model (CNN feature embeddings + XGBoost classifier). The project includes a Flask-based backend and a React (Vite) frontend for real-time predictions.

The models are trained on a curated dataset, with a structured data pipeline, feature extraction process, and optimized inference mechanism for reliable and efficient detection.

---

## Description

Users upload a face or general image; the backend returns **Real** vs **Fake** with a probability and confidence.

- **CNN baseline** — Transfer learning (e.g. MobileNetV2, EfficientNet-B0): image → sigmoid “probability of real.”
- **Hybrid model** — The same CNN’s **embedding layer** feeds **XGBoost** (with optional scaling and probability calibration).  
  **`evaluate.py`** compares CNN vs hybrid on a held-out split and writes `models/production_inference.json` to choose **`cnn_direct`** or **`hybrid`** for default API behavior.

---

## Features

- **Image upload** via web UI or `POST /api/predict`
- **Deepfake vs real** binary classification
- **Hybrid ML pipeline** (CNN + gradient-boosted classifier on features)
- **Calibrated probabilities** and validation-tuned thresholds
- **Health & model availability** endpoints for ops checks

---

## Tech Stack

| Layer | Technologies |
|--------|----------------|
| **Frontend** | React, Vite, JavaScript |
| **Backend** | Python 3.10+, Flask, Flask-CORS |
| **ML** | TensorFlow/Keras, XGBoost, scikit-learn, NumPy, Pandas, Pillow |

---

## Project Structure

```text
backend/           # Flask app (app.py, routes, services)
frontend/          # React + Vite UI
utils/             # Config, data loading, metrics, inference helpers
data/              # Place dataset here (see data/README.md) — not in Git
models/            # Trained artifacts (see models/README.md) — binaries not in Git
train_cnn.py       # CNN training
extract_features.py
train_xgboost.py
evaluate.py
predict.py         # CLI prediction
run_training.py    # Orchestrated pipeline
sanity_checks.py
verify_dataset.py
requirements.txt
```

---

## Setup

### 1. Clone

```bash
git clone https://github.com/kirti-vispute/Hybrid_Deepfake_Detector.git
cd Hybrid_Deepfake_Detector
```

### 2. Python environment

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend dependencies

```bash
npm --prefix frontend install
```

### 4. Environment (optional)

- Copy `frontend/.env.example` to `frontend/.env` if you need a non-default API URL.
- Backend dataset paths can be overridden with `HDFD_*` variables — see `utils/config.py`.

### 5. Run backend

From the **repository root**:

```bash
python backend/app.py
```

API base (default): `http://127.0.0.1:8000/api`  
Check: `GET http://127.0.0.1:8000/api/health`

### 6. Run frontend

```bash
npm --prefix frontend run dev
```

Open the printed local URL (e.g. `http://127.0.0.1:5173`).

---

## Dataset (not included)

1. Add your **`train.csv`**, **`valid.csv`**, **`test.csv`**, and image tree under **`data/`** as described in **`data/README.md`**.
2. Paths in config default to **`data/`** relative to the repo root.
3. Do **not** commit raw images or CSVs; they stay local or on your storage.

---

## Training (summary)

After the dataset is in place:

```bash
python verify_dataset.py --output-json results/dataset_audit.json
python run_training.py --mode fast
# or step-by-step: train_cnn.py → extract_features.py → train_xgboost.py → evaluate.py
```

See **`run_training.py`**, **`train_cnn.py`**, and **`evaluate.py`** for flags (fast vs strong mode, sample caps, etc.). Training produces files under **`models/`** and **`artifacts/`** (ignored by Git).

**Inference without pushing weights:** clone the repo, add data, run training once, then start the backend.

---

## API (quick reference)

- `GET /api/health` — status and model availability  
- `POST /api/predict` — multipart form: `file`, `model` (API may normalize legacy values; production routing uses `production_inference.json` when present)

---

## CLI prediction

```bash
python predict.py --image path/to/image.jpg --model hybrid
```

---

## Future improvements

- Video / frame-sequence support and temporal consistency  
- Stronger backbones and self-supervised pre-training  
- Full-test evaluation without subset caps when GPU is available  
- Docker image and CI for lint + smoke tests  
- Hosted model registry (optional) for teams that do not train locally  

---

## License

Add a `LICENSE` file if you distribute the project publicly.

---

## Security note

Never commit **`.env`**, API keys, or personal dataset paths you consider sensitive. Use env vars (`HDFD_*`) for machine-specific configuration.
