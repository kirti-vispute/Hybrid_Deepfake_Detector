from __future__ import annotations

import json
from pathlib import Path

import kagglehub
import pandas as pd


def _clean_split_csv(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if "path" not in raw.columns or "label" not in raw.columns:
        raise RuntimeError(f"Unexpected CSV format: {csv_path}")
    df = raw[["path", "label"]].copy()
    df["label"] = df["label"].astype(int)
    df["label_str"] = df["label"].map({1: "real", 0: "fake"})
    return df


def main() -> None:
    # Required exact download snippet.
    path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
    print("Path to dataset files:", path)

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(path).resolve()
    image_root = dataset_path / "real_vs_fake" / "real-vs-fake"
    train_df = _clean_split_csv(dataset_path / "train.csv")
    valid_df = _clean_split_csv(dataset_path / "valid.csv")
    test_df = _clean_split_csv(dataset_path / "test.csv")

    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    class_counts = df["label_str"].value_counts().to_dict()

    train_df.to_csv(data_dir / "train.csv", index=False)
    valid_df.to_csv(data_dir / "valid.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)

    summary = {
        "dataset_path": str(dataset_path),
        "image_root": str(image_root),
        "detected_real_dir": str(image_root / "train" / "real"),
        "detected_fake_dir": str(image_root / "train" / "fake"),
        "class_counts": class_counts,
        "split_counts": {
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
        },
        "split_class_counts": {
            "train": train_df["label_str"].value_counts().to_dict(),
            "valid": valid_df["label_str"].value_counts().to_dict(),
            "test": test_df["label_str"].value_counts().to_dict(),
        },
    }
    (results_dir / "kaggle_dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
