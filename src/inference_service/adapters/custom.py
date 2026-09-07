from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inference_service.adapters.custom_artifacts import (
    CustomConfig,
    encode,
    read_manifest,
    verify,
    version,
)
from inference_service.core.contracts import (
    AdapterTimings,
    ModelKey,
    ModelMetadata,
    Prediction,
    SentimentScores,
    TextInput,
)
from inference_service.core.exceptions import FatalWorkerError


class CustomSentimentAdapter:
    def __init__(
        self, artifact_dir: Path, *, device: str = "cpu", max_text_characters: int = 8000
    ) -> None:
        self.artifact_dir = artifact_dir
        self.device = device
        self.max_text_characters = max_text_characters
        self._files = read_manifest(artifact_dir)
        self._metadata = ModelMetadata(
            ModelKey("custom-sentiment", version(self._files)),
            "text-classification",
            "text",
            ("negative", "positive"),
            device,
            32,
        )
        self._last_timings = AdapterTimings()
        self._model: Any = None
        self._torch: Any = None
        self._indices: dict[str, int] = {}
        self._config = CustomConfig()

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def last_timings(self) -> AdapterTimings:
        return self._last_timings

    def load(self) -> None:
        verify(self.artifact_dir, self._files)
        import torch
        from safetensors.torch import load_file

        from inference_service.adapters.custom_network import SentimentNetwork

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was configured but is unavailable.")
        self._config = CustomConfig.model_validate_json(
            (self.artifact_dir / "config.json").read_text()
        )
        words = json.loads((self.artifact_dir / "vocabulary.json").read_text())
        if (
            not isinstance(words, list)
            or not all(isinstance(w, str) for w in words)
            or words[:2] != ["<pad>", "<unk>"]
            or len(set(words)) != len(words)
            or not 2 <= len(words) <= 10000
        ):
            raise RuntimeError("Invalid custom vocabulary.")
        self._indices = {word: index for index, word in enumerate(words)}
        model = SentimentNetwork(len(words), self._config)
        model.load_state_dict(
            load_file(str(self.artifact_dir / "weights.safetensors")), strict=True
        )
        model.to(self.device)
        model.eval()
        self._torch = torch
        self._model = model
        self.predict_batch([TextInput(text="warmup")])

    def validate(self, item: TextInput) -> None:
        if len(item.text) > self.max_text_characters:
            raise ValueError("Text exceeds the configured character limit.")

    def batch_key(self, item: TextInput) -> ModelKey:
        return self.metadata.key

    def predict_batch(self, items: Sequence[TextInput]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("Adapter is not loaded.")
        if not 1 <= len(items) <= self.metadata.max_batch_size:
            raise ValueError("Unsupported batch size.")
        from inference_service.adapters.custom_network import padded_batch

        start = time.perf_counter()
        for item in items:
            self.validate(item)
        try:
            batch = padded_batch(
                [encode(item.text, self._indices, self._config.max_length) for item in items],
                self.device,
            )
            if self.device == "cuda":
                self._torch.cuda.synchronize()
            forward = time.perf_counter()
            with self._torch.inference_mode():
                logits = self._model(batch)
            if self.device == "cuda":
                self._torch.cuda.synchronize()
            post = time.perf_counter()
            scores = self._torch.softmax(logits, dim=-1).cpu().tolist()
        except self._torch.cuda.OutOfMemoryError as exc:
            raise FatalWorkerError("CUDA out of memory made the worker unavailable.") from exc
        predictions = [
            Prediction(
                label="positive" if row[1] >= row[0] else "negative",
                scores=SentimentScores(negative=row[0], positive=row[1]),
            )
            for row in scores
        ]
        end = time.perf_counter()
        self._last_timings = AdapterTimings(forward - start, post - forward, end - post)
        return predictions

    def close(self) -> None:
        self._model = None
        self._indices = {}
        if self._torch is not None and self.device == "cuda":
            self._torch.cuda.empty_cache()
        self._torch = None
