from starlette.responses import JSONResponse

from inference_service.api.schemas import ErrorDetail, ErrorResponse


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def error_response(request_id: str, status: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(request_id=request_id, error=ErrorDetail(code=code, message=message))
    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
        headers={"X-Request-ID": request_id},
    )
