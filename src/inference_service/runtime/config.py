from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulingPolicy(StrEnum):
    SINGLE = "single"
    IMMEDIATE = "immediate"
    TIMED = "timed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="forbid", frozen=True)

    adapter: Literal["fake", "huggingface", "custom"] = "fake"
    artifact_dir: Path = Path("artifacts/huggingface-sst2")
    custom_artifact_dir: Path = Path("artifacts/custom-sentiment")
    device: Literal["cpu", "cuda"] = "cpu"
    max_body_bytes: int = Field(default=32 * 1024, gt=0)
    max_text_characters: int = Field(default=8000, gt=0)
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.TIMED
    max_batch_size: int = Field(default=8, gt=0)
    max_collection_delay_ms: int = Field(default=10, ge=0, le=60_000)
    pending_capacity: int = Field(default=128, gt=0)
    request_deadline_ms: int = Field(default=5_000, gt=0)
    graceful_shutdown_seconds: float = Field(default=10.0, gt=0)
    worker_watchdog_seconds: float = Field(default=30.0, gt=0)
    retry_after_seconds: int = Field(default=1, gt=0)
