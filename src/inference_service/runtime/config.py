from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="forbid", frozen=True)

    adapter: Literal["fake"] = "fake"
    max_body_bytes: int = Field(default=32 * 1024, gt=0)
    max_text_characters: int = Field(default=8000, gt=0)
