from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import shutil
import time
import uuid
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RANDOMUSER_API = "https://randomuser.me/api/?results=20&seed=deepfake-detector&inc=picture"

FAKE_BENCHMARK = [
    (f"tpdne_fake_{idx:02d}.jpg", "https://thispersondoesnotexist.com/")
    for idx in range(1, 21)
]

IMAGE_HEADERS = {"User-Agent": "Mozilla/5.0"}


def benchmark_manifest() -> list[dict]:
    entries: list[dict] = []
    randomuser_payload = _fetch_json_with_retry(RANDOMUSER_API)
    for idx, result in enumerate(randomuser_payload.get("results", []), start=1):
        source_url = str(result.get("picture", {}).get("large", "")).strip()
        if not source_url:
            continue
        filename = f"randomuser_real_{idx:02d}.jpg"
        entries.append(
            {
                "label": "real",
                "filename": filename,
                "source_name": "RandomUser",
                "source_page": RANDOMUSER_API,
                "download_url": source_url,
                "download_mode": "direct",
            }
        )
    for filename, source_url in FAKE_BENCHMARK:
        entries.append(
            {
                "label": "fake",
                "filename": filename,
                "source_name": "This Person Does Not Exist",
                "source_page": source_url,
                "download_url": source_url,
                "download_mode": "direct",
            }
        )
    return entries


def _fetch_json_with_retry(url: str, attempts: int = 5) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers=IMAGE_HEADERS)
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(12.0, 1.5 * (2 ** attempt)))
    if last_error is not None:
        raise last_error
    return {}


def _download_with_retry(url: str, target: Path, attempts: int = 5) -> None:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers=IMAGE_HEADERS)
            temp_target = target.with_suffix(target.suffix + ".part")
            with urlopen(request, timeout=60) as response, temp_target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temp_target.replace(target)
            return
        except (HTTPError, URLError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            time.sleep(min(12.0, 1.5 * (2 ** attempt)))
    if last_error is not None:
        raise last_error


def _resolve_download_url(entry: dict) -> str:
    return str(entry["download_url"])


def ensure_external_benchmark(download_root: Path) -> list[dict]:
    download_root.mkdir(parents=True, exist_ok=True)
    (download_root / "real").mkdir(parents=True, exist_ok=True)
    (download_root / "fake").mkdir(parents=True, exist_ok=True)

    downloaded: list[dict] = []
    for entry in benchmark_manifest():
        target = download_root / entry["label"] / entry["filename"]
        resolved_download_url = _resolve_download_url(entry)
        if not target.exists() or target.stat().st_size == 0:
            _download_with_retry(resolved_download_url, target)
            time.sleep(0.75)
        record = dict(entry)
        record["resolved_download_url"] = resolved_download_url
        record["local_path"] = str(target.resolve())
        downloaded.append(record)

    manifest_path = download_root / "manifest.json"
    manifest_path.write_text(json.dumps(downloaded, indent=2), encoding="utf-8")
    return downloaded


def _encode_multipart(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    file_name = file_path.name
    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts: list[bytes] = []
    for key, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def predict_file_via_api(image_path: Path, backend_url: str, model_choice: str = "hybrid") -> dict:
    endpoint = backend_url.rstrip("/") + "/api/predict"
    body, content_type = _encode_multipart({"model": model_choice}, "file", image_path)
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": content_type, "Content-Length": str(len(body))},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _compute_summary(rows: Iterable[dict]) -> tuple[dict, dict]:
    rows = list(rows)
    total = len(rows)
    real_rows = [row for row in rows if row["true_label"] == "real"]
    fake_rows = [row for row in rows if row["true_label"] == "fake"]
    pred_real = sum(1 for row in rows if row["predicted_label"] == "real")
    pred_fake = sum(1 for row in rows if row["predicted_label"] == "fake")
    fake_predicted_real = sum(1 for row in fake_rows if row["predicted_label"] == "real")
    real_predicted_fake = sum(1 for row in real_rows if row["predicted_label"] == "fake")
    correct = sum(1 for row in rows if bool(row["correct"]))
    summary = {
        "total": total,
        "real_count": len(real_rows),
        "fake_count": len(fake_rows),
        "accuracy": (correct / total) if total else 0.0,
        "fake_predicted_real": fake_predicted_real,
        "fake_to_real_error": (fake_predicted_real / len(fake_rows)) if fake_rows else 0.0,
        "real_predicted_fake": real_predicted_fake,
        "real_to_fake_error": (real_predicted_fake / len(real_rows)) if real_rows else 0.0,
        "fake_recall": 1.0 - ((fake_predicted_real / len(fake_rows)) if fake_rows else 0.0),
        "real_recall": 1.0 - ((real_predicted_fake / len(real_rows)) if real_rows else 0.0),
        "pred_real": pred_real,
        "pred_fake": pred_fake,
        "majority_ratio": max((pred_real / total) if total else 0.0, (pred_fake / total) if total else 0.0),
    }
    confusion = {
        "labels": ["fake", "real"],
        "matrix": [
            [sum(1 for row in fake_rows if row["predicted_label"] == "fake"), fake_predicted_real],
            [real_predicted_fake, sum(1 for row in real_rows if row["predicted_label"] == "real")],
        ],
    }
    return summary, confusion


def evaluate_manifest(
    manifest: list[dict],
    backend_url: str,
    reports_dir: Path,
    report_prefix: str,
    model_choice: str = "hybrid",
    hard_examples_dir: Path | None = None,
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)
    if hard_examples_dir is not None:
        hard_examples_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    hard_failures: list[dict] = []
    for entry in manifest:
        image_path = Path(entry["local_path"])
        prediction = predict_file_via_api(image_path=image_path, backend_url=backend_url, model_choice=model_choice)
        predicted_label = str(prediction.get("predicted_class", "")).strip().lower()
        normalized_pred = "real" if predicted_label == "real" else "fake"
        prob_real = float(prediction.get("probabilities", {}).get("real", 0.0))
        prob_fake = float(prediction.get("probabilities", {}).get("fake", 0.0))
        confidence = float(prediction.get("confidence", max(prob_real, prob_fake)))
        correct = normalized_pred == entry["label"]
        row = {
            "filename": image_path.name,
            "true_label": entry["label"],
            "predicted_label": normalized_pred,
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "confidence": confidence,
            "correct": bool(correct),
            "model_used": prediction.get("model_used"),
            "inference_backend": prediction.get("inference_backend"),
            "decision_threshold": prediction.get("decision_threshold"),
            "source_name": entry["source_name"],
            "source_page": entry["source_page"],
            "download_url": entry["download_url"],
            "local_path": str(image_path),
        }
        rows.append(row)

        if not correct:
            failure = dict(row)
            if hard_examples_dir is not None:
                lbl_dir = hard_examples_dir / entry["label"]
                lbl_dir.mkdir(parents=True, exist_ok=True)
                target = lbl_dir / image_path.name
                shutil.copy2(image_path, target)
                failure["hard_example_path"] = str(target.resolve())
            hard_failures.append(failure)

    csv_path = reports_dir / f"{report_prefix}_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["filename"])
        writer.writeheader()
        writer.writerows(rows)

    summary, confusion = _compute_summary(rows)
    summary_path = reports_dir / f"{report_prefix}_metrics.json"
    confusion_path = reports_dir / f"{report_prefix}_confusion_matrix.json"
    hard_failures_path = reports_dir / f"{report_prefix}_hard_failures.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    confusion_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    hard_failures_path.write_text(json.dumps(hard_failures, indent=2), encoding="utf-8")

    return {
        "predictions_csv": str(csv_path.resolve()),
        "metrics_json": str(summary_path.resolve()),
        "confusion_json": str(confusion_path.resolve()),
        "hard_failures_json": str(hard_failures_path.resolve()),
        "summary": summary,
        "hard_failures": hard_failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and evaluate a small external real/fake face benchmark.")
    parser.add_argument("--download-root", type=str, default="external_test")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--backend-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=str, default="hybrid")
    parser.add_argument("--hard-examples-dir", type=str, default="data/hard_examples/fake/external")
    parser.add_argument("--report-prefix", type=str, default="external_benchmark")
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ensure_external_benchmark(Path(args.download_root))
    if args.download_only:
        print(json.dumps({"downloaded": len(manifest), "manifest": str((Path(args.download_root) / "manifest.json").resolve())}, indent=2))
        return

    result = evaluate_manifest(
        manifest=manifest,
        backend_url=args.backend_url,
        reports_dir=Path(args.reports_dir),
        report_prefix=args.report_prefix,
        model_choice=args.model,
        hard_examples_dir=Path(args.hard_examples_dir),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
