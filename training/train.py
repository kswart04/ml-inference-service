"""Train from initialization using training and validation splits only."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from inference_service.adapters.custom_artifacts import (
    FILES,
    CustomConfig,
    encode,
    sha256,
    version,
    vocabulary,
    write_json,
)
from inference_service.adapters.custom_network import SentimentNetwork, padded_batch
from training.data import Row, distribution, read_split
from training.metrics import classification_metrics


def predict(model: SentimentNetwork, sequences: list[list[int]]) -> list[int]:
    model.eval()
    result: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 32):
            result.extend(model(padded_batch(sequences[start : start + 32])).argmax(-1).tolist())
    return result


def small_subset(rows: list[Row], seed: int) -> list[Row]:
    # A deterministic balanced sample, independent of validation and test labels.
    rng = random.Random(seed)
    result: list[Row] = []
    for domain in sorted({r["domain"] for r in rows}):
        for label in (0, 1):
            group = [r for r in rows if r["domain"] == domain and r["label"] == label]
            rng.shuffle(group)
            result.extend(group[:200])
    rng.shuffle(result)
    return result


def train(dataset: Path, output: Path, profile: str) -> None:
    if output.exists():
        raise ValueError("Output already exists; use a new experiment directory.")
    started = time.perf_counter()
    config = CustomConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_rows = read_split(dataset, "train")
    validation_rows = read_split(dataset, "validation")
    if profile == "small":
        train_rows = small_subset(train_rows, config.seed)
    words = vocabulary((r["text"] for r in train_rows), config.max_length)
    indices = {word: index for index, word in enumerate(words)}
    sequences = [encode(r["text"], indices, config.max_length) for r in train_rows]
    validation = [encode(r["text"], indices, config.max_length) for r in validation_rows]
    labels = torch.tensor([r["label"] for r in train_rows])
    validation_labels = [r["label"] for r in validation_rows]
    model = SentimentNetwork(len(words), config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    epochs = 25 if profile == "small" else 50
    history = []
    best_score, best_epoch = -1.0, 0
    best_weights = copy.deepcopy(model.state_dict())
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(sequences), generator=generator).tolist()
        total_loss = 0.0
        for start in range(0, len(order), 32):
            positions = order[start : start + 32]
            batch = padded_batch([sequences[i] for i in positions])
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch), labels[positions])
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total_loss += loss.item() * len(positions)
        metrics = classification_metrics(validation_labels, predict(model, validation))
        score = float(str(metrics["macro_f1"]))
        history.append(
            {"epoch": epoch, "train_loss": total_loss / len(sequences), "validation": metrics}
        )
        print(f"epoch={epoch} validation_macro_f1={score:.4f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_weights = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 7:
            break
    model.load_state_dict(best_weights)
    train_metrics = classification_metrics(labels.tolist(), predict(model, sequences))
    val_metrics = classification_metrics(validation_labels, predict(model, validation))
    majority = int(sum(labels.tolist()) > len(labels) / 2)
    baseline = classification_metrics(validation_labels, [majority] * len(validation_labels))
    if float(str(val_metrics["accuracy"])) <= float(str(baseline["accuracy"])):
        raise RuntimeError("Validation accuracy does not beat the training-majority baseline.")
    output.mkdir(parents=True)
    write_json(output / "config.json", config.model_dump(mode="json"))
    write_json(output / "vocabulary.json", words)
    save_file(dict(sorted(best_weights.items())), str(output / "weights.safetensors"))
    source_paths = (
        sorted(Path("training").glob("*.py"))
        + sorted(Path("src/inference_service/adapters").glob("custom*.py"))
        + [Path("uv.lock")]
    )
    report = {
        "profile": profile,
        "seed": config.seed,
        "device": "cpu",
        "precision": "float32",
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "max_epochs": epochs,
        "batch_size": 32,
        "learning_rate": 0.003,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "early_stopping_patience": 7,
        "selection": "maximum validation macro-F1; earliest epoch wins ties",
        "best_epoch": best_epoch,
        "vocabulary_size": len(words),
        "vocabulary_fit": "selected training rows only",
        "test_accessed": False,
        "train_ids": [r["id"] for r in train_rows],
        "validation_ids": [r["id"] for r in validation_rows],
        "training_distribution": distribution(train_rows),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "dataset": json.loads((dataset / "manifest.json").read_text()),
        "majority_label": majority,
        "validation_majority_baseline": baseline,
        "train": train_metrics,
        "validation": val_metrics,
        "history": history,
        "training_wall_seconds": time.perf_counter() - started,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "processor": platform.processor(),
            "cuda_available": torch.cuda.is_available(),
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for p in source_paths},
    }
    write_json(output / "training.json", report)
    files = {name: sha256(output / name) for name in sorted(FILES)}
    write_json(
        output / "manifest.json", {"schema_version": 1, "version": version(files), "files": files}
    )
    print(
        json.dumps(
            {
                "version": version(files),
                "validation": val_metrics,
                "seconds": report["training_wall_seconds"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/uci-sentiment"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/custom-sentiment"))
    parser.add_argument("--profile", choices=("small", "full"), default="small")
    args = parser.parse_args()
    train(args.dataset, args.output, args.profile)


if __name__ == "__main__":
    main()
