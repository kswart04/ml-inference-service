from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from inference_service.api.app import create_app
from inference_service.runtime.config import Settings


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings()), raise_server_exceptions=False) as test_client:
        yield test_client
