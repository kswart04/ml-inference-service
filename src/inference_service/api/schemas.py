from pydantic import BaseModel, ConfigDict, Field

from inference_service.core.contracts import Prediction, TextInput


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    input: TextInput


class PredictResponse(BaseModel):
    request_id: str
    model_id: str
    model_version: str
    prediction: Prediction


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    request_id: str
    error: ErrorDetail


class ModelInfo(BaseModel):
    model_id: str
    model_version: str
    task: str
    input_type: str
    labels: tuple[str, ...]
    device: str
    max_batch_size: int
    available: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
