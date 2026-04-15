# Model artifacts (not in Git)

Binary weights and calibrators are **gitignored**. After you train locally, this folder should contain files such as:

- `cnn_baseline.keras` — fine-tuned CNN
- `cnn_calibrator.joblib` — probability calibration for the CNN
- `cnn_metadata.json` — backbone, image size, decision threshold
- `xgboost_hybrid.joblib` — hybrid classifier on CNN embeddings
- `hybrid_feature_scaler.joblib` — feature scaling for XGBoost
- `hybrid_calibrator.joblib` — hybrid probability calibration
- `hybrid_metadata.json` — hybrid threshold and training metadata
- `smart_router.json` — optional fusion router (if enabled)
- `production_inference.json` — written by `evaluate.py` (`cnn_direct` vs `hybrid`)

See the root **README.md** for training commands. Paths are resolved relative to the project root (or via `HDFD_*` env vars in `utils/config.py`).
