"""Evaluate a frozen export once on its held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inference_service.adapters.custom import CustomSentimentAdapter
from inference_service.adapters.custom_artifacts import sha256, write_json
from inference_service.core.contracts import TextInput
from training.data import read_split
from training.metrics import classification_metrics


def evaluate(dataset: Path, artifact: Path, output: Path) -> None:
    if output.exists() or (artifact / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; do not tune against this test split.")
    adapter = CustomSentimentAdapter(artifact)
    adapter.load()
    try:
        training = json.loads((artifact / "training.json").read_text())
        if sha256(dataset / "manifest.json") != training["dataset_manifest_sha256"]:
            raise ValueError("Evaluation dataset differs from training provenance.")
        rows = read_split(dataset, "test")
        if {r["id"] for r in rows} & set(training["train_ids"] + training["validation_ids"]):
            raise ValueError("Test IDs overlap fitting or selection data.")
        labels = [r["label"] for r in rows]
        predictions: list[int] = []
        for start in range(0, len(rows), 32):
            results = adapter.predict_batch(
                [TextInput(text=r["text"]) for r in rows[start : start + 32]]
            )
            predictions.extend(int(r.label == "positive") for r in results)
        metrics = classification_metrics(labels, predictions)
        baseline = classification_metrics(labels, [training["majority_label"]] * len(labels))
        passed = float(str(metrics["accuracy"])) > float(str(baseline["accuracy"])) and float(
            str(metrics["macro_f1"])
        ) > float(str(baseline["macro_f1"]))
        report = {
            "model_version": adapter.metadata.key.model_version,
            "artifact_files": adapter._files,
            "profile": training["profile"],
            "seed": training["seed"],
            "dataset": training["dataset"],
            "best_epoch": training["best_epoch"],
            "vocabulary_size": training["vocabulary_size"],
            "training_distribution": training["training_distribution"],
            "validation": training["validation"],
            "test": metrics,
            "training_majority_test_baseline": baseline,
            "release_gate_passed": passed,
            "training_wall_seconds": training["training_wall_seconds"],
            "hardware": training["hardware"],
            "versions": training["versions"],
            "source_sha256": training["source_sha256"],
            "git_base": training["git_base"],
        }
        write_json(artifact / "evaluation.json", report)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        print(json.dumps(report, indent=2))
        if not passed:
            raise RuntimeError("Held-out baseline gate failed. Investigate; do not claim release.")
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/uci-sentiment"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/custom-sentiment"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/custom-small.json"))
    args = parser.parse_args()
    evaluate(args.dataset, args.artifact, args.output)


if __name__ == "__main__":
    main()
