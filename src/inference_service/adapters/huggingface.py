from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inference_service.core.contracts import (
    AdapterTimings,
    ModelKey,
    ModelMetadata,
    Prediction,
    SentimentScores,
    TextInput,
)
from inference_service.core.exceptions import FatalWorkerError

MODEL_REPO_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
MODEL_INTERNAL_VERSION = "hf-sst2-714eb0fa-max256"
MODEL_LICENSE = "apache-2.0"
MODEL_MAX_LENGTH = 256
MANIFEST_NAME = "inference-service-manifest.json"
REQUIRED_FILES = frozenset(
    {"README.md", "config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HuggingFaceSentimentAdapter:
    """Load the pinned DistilBERT SST-2 model from local files."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        device: str = "cpu",
        max_text_characters: int = 8000,
        max_batch_size: int = 32,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.device = device
        self.max_text_characters = max_text_characters
        self._metadata = ModelMetadata(
            key=ModelKey("huggingface-sentiment", MODEL_INTERNAL_VERSION),
            task="text-classification",
            input_type="text",
            labels=("negative", "positive"),
            device=device,
            max_batch_size=max_batch_size,
        )
        self._last_timings = AdapterTimings()
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def last_timings(self) -> AdapterTimings:
        return self._last_timings

    def _validate_manifest(self) -> None:
        manifest_path = self.artifact_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing prepared artifact manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Prepared artifact manifest is unreadable.") from exc
        expected = {
            "schema_version": 1,
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "internal_version": MODEL_INTERNAL_VERSION,
            "license": MODEL_LICENSE,
            "max_length": MODEL_MAX_LENGTH,
        }
        if any(manifest.get(name) != value for name, value in expected.items()):
            raise RuntimeError("Prepared artifact manifest does not match the configured model.")
        files = manifest.get("files")
        if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
            raise RuntimeError("Prepared artifact manifest has an unexpected file set.")
        for name, expected_hash in files.items():
            path = self.artifact_dir / name
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RuntimeError(f"Prepared artifact failed integrity check: {name}")

    def load(self) -> None:
        self._validate_manifest()
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'hf' dependency extra to use this adapter.") from exc
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was configured but is unavailable.")
        tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        if model.config.num_labels != 2:
            raise RuntimeError("Expected a two-label sequence-classification model.")
        normalized = {
            int(index): str(label).upper() for index, label in model.config.id2label.items()
        }
        if normalized != {0: "NEGATIVE", 1: "POSITIVE"}:
            raise RuntimeError("Model label mapping is incompatible with the response schema.")
        if tokenizer.model_max_length < MODEL_MAX_LENGTH:
            raise RuntimeError("Tokenizer maximum length is below the configured limit.")
        model.to(self.device)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self.predict_batch([TextInput(text="warmup")])

    def validate(self, item: TextInput) -> None:
        if len(item.text) > self.max_text_characters:
            raise ValueError("Text exceeds the configured character limit.")

    def batch_key(self, item: TextInput) -> ModelKey:
        return self.metadata.key

    def predict_batch(self, items: Sequence[TextInput]) -> list[Prediction]:
        if self._model is None or self._tokenizer is None or self._torch is None:
            raise RuntimeError("Adapter is not loaded.")
        if not 1 <= len(items) <= self.metadata.max_batch_size:
            raise ValueError("Unsupported batch size.")
        for item in items:
            self.validate(item)

        preprocess_started = time.perf_counter()
        encoded = self._tokenizer(
            [item.text for item in items],
            padding="longest",
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        forward_started = time.perf_counter()
        try:
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
            if self.device == "cuda":
                self._torch.cuda.synchronize()
        except self._torch.cuda.OutOfMemoryError as exc:
            raise FatalWorkerError("CUDA out of memory made the worker unavailable.") from exc
        postprocess_started = time.perf_counter()
        probabilities = self._torch.softmax(logits, dim=-1).detach().cpu().tolist()
        results = [
            Prediction(
                label="positive" if scores[1] >= scores[0] else "negative",
                scores=SentimentScores(negative=scores[0], positive=scores[1]),
            )
            for scores in probabilities
        ]
        finished = time.perf_counter()
        self._last_timings = AdapterTimings(
            preprocessing_seconds=forward_started - preprocess_started,
            forward_seconds=postprocess_started - forward_started,
            postprocessing_seconds=finished - postprocess_started,
        )
        return results

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()
        self._torch = None
