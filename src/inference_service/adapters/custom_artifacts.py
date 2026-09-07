"""Versioned, framework-independent text and artifact contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FILES = frozenset({"config.json", "vocabulary.json", "weights.safetensors", "training.json"})


class CustomConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    architecture: Literal["embedding-masked-mean-mlp-v1"] = "embedding-masked-mean-mlp-v1"
    tokenizer: Literal["lowercase-ascii-words-apostrophes-v1"] = (
        "lowercase-ascii-words-apostrophes-v1"
    )
    labels: tuple[Literal["negative"], Literal["positive"]] = ("negative", "positive")
    embedding_dim: int = Field(default=64, gt=0, le=512)
    hidden_dim: int = Field(default=64, gt=0, le=512)
    max_length: int = Field(default=256, gt=0, le=2048)
    seed: int = 42


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def vocabulary(texts: Iterable[str], max_length: int = 256) -> list[str]:
    counts = Counter(token for text in texts for token in tokens(text)[:max_length])
    return ["<pad>", "<unk>"] + sorted(counts, key=lambda word: (-counts[word], word))[:9998]


def encode(text: str, indices: dict[str, int], max_length: int) -> list[int]:
    return [indices.get(token, 1) for token in tokens(text)[:max_length]] or [1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def version(files: dict[str, str]) -> str:
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return "custom-" + digest[:16]


def read_manifest(directory: Path) -> dict[str, str]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        files = manifest["files"]
        if (
            manifest["schema_version"] != 1
            or not isinstance(files, dict)
            or set(files) != FILES
            or any(
                not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in files.values()
            )
            or manifest["version"] != version(files)
        ):
            raise ValueError("Invalid manifest")
        return {str(k): str(v) for k, v in files.items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Missing or invalid custom artifact manifest in {directory}") from exc


def verify(directory: Path, files: dict[str, str]) -> None:
    for name, digest in files.items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise RuntimeError(f"Custom artifact integrity check failed: {name}")
