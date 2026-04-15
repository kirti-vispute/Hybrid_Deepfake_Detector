# Dataset (not included in this repository)

Place your **Deepfake vs Real** dataset here after cloning. Do not commit images or CSVs.

## Expected layout

```text
data/
  real_vs_fake/          # or nested: real_vs_fake/real-vs-fake/ — the loader auto-resolves
  train.csv
  valid.csv
  test.csv
```

## CSV columns

- **`path`** — relative image path (used for loading)
- **`label`** — `0` = fake, `1` = real
- **`label_str`** — human-readable class name

`original_path` (if present) is ignored at runtime.

## Override paths

Set environment variables (see root `README.md`) or edit `utils/config.py` defaults — for example `HDFD_DATASET_ROOT` / `HDFD_TRAIN_CSV`.
