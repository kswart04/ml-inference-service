"""Typed contracts shared by the API, registry, and adapters."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str

    @field_validator("text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain a non-whitespace character.")
        return value


class SentimentScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    negative: float = Field(ge=0, le=1)
    positive: float = Field(ge=0, le=1)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Literal["negative", "positive"]
    scores: SentimentScores


@dataclass(frozen=True)
class ModelKey:
    model_id: str
    model_version: str


@dataclass(frozen=True)
class ModelMetadata:
    key: ModelKey
    task: str
    input_type: str
    labels: tuple[str, ...]
    device: str
    max_batch_size: int


@dataclass(frozen=True)
class AdapterTimings:
    preprocessing_seconds: float = 0.0
    forward_seconds: float = 0.0
    postprocessing_seconds: float = 0.0
