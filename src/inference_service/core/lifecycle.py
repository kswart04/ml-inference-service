from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from inference_service.core.contracts import Prediction, TextInput


class RequestState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {RequestState.SUCCEEDED, RequestState.FAILED, RequestState.EXPIRED, RequestState.CANCELLED}
)


@dataclass(slots=True)
class RequestEnvelope:
    request_id: str
    item: TextInput
    request_started_at: float
    admitted_at: float
    deadline_at: float
    future: asyncio.Future[Prediction]
    state: RequestState = field(default=RequestState.PENDING)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES
