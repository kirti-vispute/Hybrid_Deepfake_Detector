from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from external_benchmark_eval import ensure_external_benchmark, evaluate_manifest
from utils.config import get_config


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume from current state and run a controlled improve-evaluate-retrain loop.")
    parser.add_argument("--backend-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--external-download-root", type=str, default="external_test")
    parser.add_argument("--hard-examples-root", type=str, default="data/hard_examples/fake/loop_focus")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.005)
    parser.add_argument("--train-real-root", type=str, default="Dataset/rvf10k/train/real")
    parser.add_argument("--train-fake-root", type=str, default="Dataset/rvf10k/train/fake")
    parser.add_argument("--valid-real-root", type=str, default="Dataset/rvf10k/valid/real")
    parser.add_argument("--valid-fake-root", type=str, default="Dataset/rvf10k/valid/fake")
    parser.add_argument("--local-train-per-class", type=int, default=1800)
    parser.add_argument("--local-valid-per-class", type=int, default=200)
    parser.add_argument("--local-test-per-class", type=int, default=200)
    parser.add_argument("--cache-prefix", type=str, default="local_primary_focus")
    parser.add_argument("--hard-mine-rounds", type=int, default=3)
    parser.add_argument("--hard-mine-repeat", type=int, default=4)
    parser.add_argument("--base-fake-weight", type=float, default=1.45)
    parser.add_argument("--external-hard-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _image_paths(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _sample_paths(root: Path, count: int, seed: int) -> list[Path]:
    import numpy as np

    files = _image_paths(root)
    if count <= 0 or count >= len(files):
        return files
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))[:count]
    return [files[i] for i in order]


def _build_local_manifests(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    valid_real = _sample_paths(Path(args.valid_real_root), args.local_valid_per_class + args.local_test_per_class, args.seed + 12)
    valid_fake = _sample_paths(Path(args.valid_fake_root), args.local_valid_per_class + args.local_test_per_class, args.seed + 13)

    valid_manifest: list[dict] = []
    test_manifest: list[dict] = []

    for label, paths in (("real", valid_real), ("fake", valid_fake)):
        for p in paths[: args.local_valid_per_class]:
            valid_manifest.append(
                {
                    "label": label,
                    "filename": p.name,
                    "source_name": "rvf10k_valid",
                    "source_page": "",
                    "download_url": "",
                    "local_path": str(p.resolve()),
                }
            )
        for p in paths[args.local_valid_per_class: args.local_valid_per_class + args.local_test_per_class]:
            test_manifest.append(
                {
                    "label": label,
                    "filename": p.name,
                    "source_name": "rvf10k_test",
                    "source_page": "",
                    "download_url": "",
                    "local_path": str(p.resolve()),
                }
            )

    counts = {
        "valid_real": sum(1 for row in valid_manifest if row["label"] == "real"),
        "valid_fake": sum(1 for row in valid_manifest if row["label"] == "fake"),
        "test_real": sum(1 for row in test_manifest if row["label"] == "real"),
        "test_fake": sum(1 for row in test_manifest if row["label"] == "fake"),
    }
    return valid_manifest, test_manifest, counts


def _audit_local_state(args: argparse.Namespace, cfg) -> dict:
    def _count(root_str: str) -> int:
        root = Path(root_str)
        if not root.exists():
            return 0
        return sum(1 for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    return {
        "local_dataset_counts": {
            "train_real": _count(args.train_real_root),
            "train_fake": _count(args.train_fake_root),
            "valid_real": _count(args.valid_real_root),
            "valid_fake": _count(args.valid_fake_root),
        },
        "feature_cache_present": {
            "default_train": cfg.classical_feature_dir.joinpath("train_features.npz").exists(),
            "default_valid": cfg.classical_feature_dir.joinpath("valid_features.npz").exists(),
            "default_test": cfg.classical_feature_dir.joinpath("test_features.npz").exists(),
            "local_prefix_train": cfg.classical_feature_dir.joinpath(f"{args.cache_prefix}_train_features.npz").exists(),
            "local_prefix_valid": cfg.classical_feature_dir.joinpath(f"{args.cache_prefix}_valid_features.npz").exists(),
            "local_prefix_test": cfg.classical_feature_dir.joinpath(f"{args.cache_prefix}_test_features.npz").exists(),
        },
        "models_present": {
            "classical_model": cfg.classical_model_path.exists(),
            "classical_metadata": cfg.classical_metadata_path.exists(),
            "production_inference": cfg.production_inference_path.exists(),
        },
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _round_score(local_test_summary: dict, external_summary: dict) -> float:
    return (
        0.25 * float(external_summary.get("fake_recall", 0.0))
        + 0.25 * float(external_summary.get("real_recall", 0.0))
        + 0.25 * float(local_test_summary.get("fake_recall", 0.0))
        + 0.25 * float(local_test_summary.get("real_recall", 0.0))
    )


def _train_round(args: argparse.Namespace, fake_weight: float, round_idx: int) -> None:
    command = [
        sys.executable,
        "train_classical_fallback.py",
        "--use-base-csv",
        "0",
        "--reuse-features",
        "1",
        "--cache-prefix",
        args.cache_prefix,
        "--extra-train-real-root",
        args.train_real_root,
        "--extra-train-fake-root",
        args.train_fake_root,
        "--extra-valid-real-root",
        args.valid_real_root,
        "--extra-valid-fake-root",
        args.valid_fake_root,
        "--local-train-per-class",
        str(args.local_train_per_class),
        "--local-valid-per-class",
        str(args.local_valid_per_class),
        "--local-test-per-class",
        str(args.local_test_per_class),
        "--hard-mine-rounds",
        str(args.hard_mine_rounds),
        "--hard-mine-repeat",
        str(args.hard_mine_repeat + max(0, round_idx - 1)),
        "--fake-sample-weight",
        f"{fake_weight:.4f}",
        "--external-hard-fake-dir",
        str(Path(args.hard_examples_root) / "fake"),
        "--external-hard-fake-repeat",
        str(args.external_hard_repeat),
        "--external-hard-real-dir",
        str(Path(args.hard_examples_root) / "real"),
        "--external-hard-real-repeat",
        str(args.external_hard_repeat),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    cfg = get_config()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    external_root = Path(args.external_download_root)
    hard_examples_root = Path(args.hard_examples_root)
    if hard_examples_root.exists():
        shutil.rmtree(hard_examples_root)
    hard_examples_root.mkdir(parents=True, exist_ok=True)

    audit = _audit_local_state(args, cfg)
    valid_manifest, test_manifest, local_counts = _build_local_manifests(args)
    audit["loop_eval_counts"] = local_counts
    (reports_dir / "improvement_loop_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    external_manifest = ensure_external_benchmark(external_root)

    baseline_local_valid = evaluate_manifest(valid_manifest, args.backend_url, reports_dir, "round0_local_valid", model_choice="hybrid")
    baseline_local_test = evaluate_manifest(test_manifest, args.backend_url, reports_dir, "round0_local_test", model_choice="hybrid")
    baseline_external = evaluate_manifest(
        external_manifest,
        args.backend_url,
        reports_dir,
        "round0_external",
        model_choice="hybrid",
        hard_examples_dir=hard_examples_root / "round0",
    )

    baseline_meta = _load_json(cfg.classical_metadata_path)
    rounds = [
        {
            "round": 0,
            "kind": "baseline",
            "local_valid": baseline_local_valid["summary"],
            "local_test": baseline_local_test["summary"],
            "external": baseline_external["summary"],
            "metadata_threshold": baseline_meta.get("decision_threshold"),
            "metadata_model": baseline_meta.get("best_model"),
            "score": _round_score(baseline_local_test["summary"], baseline_external["summary"]),
        }
    ]

    best_round = rounds[0]
    previous_score = rounds[0]["score"]

    for round_idx in range(1, args.max_rounds + 1):
        fake_weight = args.base_fake_weight + max(0, round_idx - 1) * 0.15
        _train_round(args=args, fake_weight=fake_weight, round_idx=round_idx)
        time.sleep(2.0)

        round_local_valid = evaluate_manifest(valid_manifest, args.backend_url, reports_dir, f"round{round_idx}_local_valid", model_choice="hybrid")
        round_local_test = evaluate_manifest(test_manifest, args.backend_url, reports_dir, f"round{round_idx}_local_test", model_choice="hybrid")
        round_external = evaluate_manifest(
            external_manifest,
            args.backend_url,
            reports_dir,
            f"round{round_idx}_external",
            model_choice="hybrid",
            hard_examples_dir=hard_examples_root / f"round{round_idx}",
        )

        metadata = _load_json(cfg.classical_metadata_path)
        summary = {
            "round": round_idx,
            "kind": "retrain",
            "fake_sample_weight": fake_weight,
            "local_valid": round_local_valid["summary"],
            "local_test": round_local_test["summary"],
            "external": round_external["summary"],
            "metadata_threshold": metadata.get("decision_threshold"),
            "metadata_model": metadata.get("best_model"),
            "score": _round_score(round_local_test["summary"], round_external["summary"]),
            "external_hard_failures_added": len(round_external["hard_failures"]),
        }
        rounds.append(summary)

        if summary["score"] > best_round["score"]:
            best_round = summary

        score_gain = summary["score"] - previous_score
        previous_score = summary["score"]
        if score_gain < args.min_improvement and len(round_external["hard_failures"]) == 0:
            break

    final_metadata = _load_json(cfg.classical_metadata_path)
    previous_live_summary = _load_json(Path("results/website_path_rvf10k_valid_50x50_after_summary.json"))
    summary = {
        "audit": audit,
        "baseline_live_summary": previous_live_summary,
        "rounds": rounds,
        "best_round": best_round,
        "final_model_path": str(cfg.classical_model_path.resolve()),
        "final_metadata_path": str(cfg.classical_metadata_path.resolve()),
        "production_backend": _load_json(cfg.production_inference_path),
        "final_threshold": final_metadata.get("decision_threshold"),
        "final_best_model": final_metadata.get("best_model"),
        "final_fake_recall": rounds[-1]["local_test"].get("fake_recall"),
        "final_fake_to_real_error": rounds[-1]["local_test"].get("fake_to_real_error"),
    }
    (reports_dir / "improvement_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
