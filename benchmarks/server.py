"""Run a benchmark server with fixed CPU thread counts."""

import os

import uvicorn


def main() -> None:
    if os.environ.get("INFERENCE_ADAPTER", "fake") != "fake":
        import torch

        torch.set_num_threads(int(os.environ.get("BENCH_TORCH_THREADS", "2")))
        torch.set_num_interop_threads(1)
    uvicorn.run(
        "inference_service.api.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=int(os.environ.get("BENCH_PORT", "8765")),
        workers=1,
        access_log=False,
    )


if __name__ == "__main__":
    main()
