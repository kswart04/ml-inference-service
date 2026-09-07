from collections.abc import Sequence
from typing import Protocol

from inference_service.core.contracts import ModelKey, ModelMetadata, Prediction, TextInput


class ModelAdapter(Protocol):
    """Blocking contract: real adapters will be invoked in M1's execution thread.

    predict_batch returns one ordered result per input. Real model adapters must
    execute one batched forward pass. The fake adapter has no neural network.
    """

    @property
    def metadata(self) -> ModelMetadata: ...

    def load(self) -> None: ...

    def validate(self, item: TextInput) -> None: ...

    def batch_key(self, item: TextInput) -> ModelKey: ...

    def predict_batch(self, items: Sequence[TextInput]) -> list[Prediction]: ...

    def close(self) -> None: ...
