"""A reproducible test fixture, not a trained sentiment model."""

import re
from collections.abc import Sequence

from inference_service.core.contracts import (
    ModelKey,
    ModelMetadata,
    Prediction,
    SentimentScores,
    TextInput,
)

POSITIVE_WORDS = frozenset({"excellent", "good", "great", "love", "wonderful"})
NEGATIVE_WORDS = frozenset({"awful", "bad", "hate", "poor", "terrible"})


class FakeAdapter:
    def __init__(self, *, max_text_characters: int = 8000) -> None:
        self._max_text_characters = max_text_characters
        self._loaded = False
        self._metadata = ModelMetadata(
            key=ModelKey("fake-sentiment", "v1"),
            task="text-classification",
            input_type="text",
            labels=("negative", "positive"),
            device="cpu",
            max_batch_size=8,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def load(self) -> None:
        self._loaded = True

    def validate(self, item: TextInput) -> None:
        if len(item.text) > self._max_text_characters:
            raise ValueError("Text exceeds the configured character limit.")

    def batch_key(self, item: TextInput) -> ModelKey:
        return self.metadata.key

    def predict_batch(self, items: Sequence[TextInput]) -> list[Prediction]:
        if not self._loaded:
            raise RuntimeError("Adapter is not loaded.")
        if not 1 <= len(items) <= self.metadata.max_batch_size:
            raise ValueError("Unsupported batch size.")
        results = []
        for item in items:
            self.validate(item)
            words = re.findall(r"[a-z]+", item.text.lower())
            balance = sum((word in POSITIVE_WORDS) - (word in NEGATIVE_WORDS) for word in words)
            positive = 0.8 if balance > 0 else 0.2 if balance < 0 else 0.5
            results.append(
                Prediction(
                    label="positive" if positive >= 0.5 else "negative",
                    scores=SentimentScores(negative=1.0 - positive, positive=positive),
                )
            )
        return results

    def close(self) -> None:
        self._loaded = False
