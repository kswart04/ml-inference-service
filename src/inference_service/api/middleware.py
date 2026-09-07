"""Bound raw HTTP bodies before JSON parsing and assign server request IDs."""

from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from inference_service.api.errors import error_response


class RequestBoundaryMiddleware:
    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > self.max_body_bytes:
                response = error_response(
                    request_id, 413, "body_too_large", "HTTP body exceeds the configured limit."
                )
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        consumed = False

        async def bounded_receive() -> Message:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        async def traced_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ]
                message["headers"] = [*headers, (b"x-request-id", request_id.encode("ascii"))]
            await send(message)

        await self.app(scope, bounded_receive, traced_send)
