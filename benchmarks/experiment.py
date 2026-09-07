"""Launch one local server per policy and retain reproducible experiment evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil

from benchmarks.load import TEXTS, run_load
from inference_service.adapters.custom_artifacts import sha256, write_json


def source_hashes() -> dict[str, str]:
    return {
        str(p): sha256(p)
        for directory in ("benchmarks", "src")
        for p in sorted(Path(directory).rglob("*.py"))
    }


async def sample_resources(
    server: psutil.Process, stop: asyncio.Event, samples: list[dict[str, float]]
) -> None:
    client_process = psutil.Process()
    server.cpu_percent()
    client_process.cpu_percent()
    while not stop.is_set():
        samples.append(
            {
                "rss_bytes": float(server.memory_info().rss),
                "cpu_percent": server.cpu_percent(),
                "client_cpu_percent": client_process.cpu_percent(),
                "client_rss_bytes": float(client_process.memory_info().rss),
            }
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except TimeoutError:
            pass


async def experiment(args: argparse.Namespace) -> None:
    output: Path = args.output
    initial_source_hashes = await asyncio.to_thread(source_hashes)
    initial_lock_hash = sha256(Path("uv.lock"))
    if await asyncio.to_thread(output.exists):
        raise ValueError("Use a fresh output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("benchmarks/raw") / output.stem
    raw_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    for policy in args.policies:
        env = {
            **os.environ,
            "INFERENCE_ADAPTER": args.adapter,
            "INFERENCE_DEVICE": "cpu",
            "INFERENCE_SCHEDULING_POLICY": policy,
            "INFERENCE_MAX_BATCH_SIZE": str(args.batch_size),
            "INFERENCE_MAX_COLLECTION_DELAY_MS": str(args.window_ms),
            "INFERENCE_PENDING_CAPACITY": str(args.pending_capacity),
            "INFERENCE_REQUEST_DEADLINE_MS": "2000",
            "BENCH_PORT": str(args.port),
            "BENCH_TORCH_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        with (raw_dir / f"{policy}-server.log").open("w") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "benchmarks.server", env=env, stdout=log, stderr=log
            )
            try:
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{args.port}",
                    timeout=5.0,
                    limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
                ) as client:
                    for _ in range(600):
                        if process.returncode is not None:
                            raise RuntimeError(f"Server exited; see {log.name}")
                        try:
                            if (await client.get("/health/ready")).status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        await asyncio.sleep(0.1)
                    else:
                        raise RuntimeError("Server readiness timed out")
                    models = (await client.get("/v1/models")).json()["models"]
                    identity = {k: models[0][k] for k in ("model_id", "model_version")}
                    for _ in range(2):
                        responses = await asyncio.gather(
                            *[
                                client.post(
                                    "/v1/predict",
                                    json={**identity, "input": {"text": TEXTS[i % len(TEXTS)]}},
                                )
                                for i in range(args.batch_size)
                            ]
                        )
                        for response in responses:
                            response.raise_for_status()
                    server = psutil.Process(process.pid)
                    for repetition in range(args.repetitions):
                        for rate in args.rates:
                            for burst in [False, True] if args.bursts else [False]:
                                samples: list[dict[str, float]] = []
                                stop = asyncio.Event()

                                monitor = asyncio.create_task(
                                    sample_resources(server, stop, samples)
                                )
                                try:
                                    name = f"{policy}-{repetition}-{rate}-{burst}"
                                    result = await run_load(
                                        client,
                                        identity,
                                        rate=rate,
                                        duration=args.duration,
                                        burst=burst,
                                        raw_path=raw_dir / f"{name}.jsonl.gz",
                                    )
                                finally:
                                    stop.set()
                                    await monitor
                                result.update(
                                    {
                                        "policy": policy,
                                        "repetition": repetition,
                                        "model": identity,
                                        "resources": samples,
                                    }
                                )
                                results.append(result)
                                print(
                                    json.dumps(
                                        {
                                            "policy": policy,
                                            "rate": rate,
                                            "burst": burst,
                                            "rep": repetition,
                                            "outcomes": result["outcomes"],
                                            "p95": result["success_latency_seconds"]["p95"],
                                            "client_valid": result["client_valid"],
                                        }
                                    ),
                                    flush=True,
                                )
                                write_json(
                                    output,
                                    {
                                        "config": vars(args) | {"output": str(output)},
                                        "results": results,
                                    },
                                )
            finally:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=15)
                except TimeoutError:
                    process.kill()
                    await process.wait()
    manifest = Path("artifacts") / (
        "custom-sentiment/manifest.json"
        if args.adapter == "custom"
        else "huggingface-sst2/inference-service-manifest.json"
    )
    provenance = {
        "git_commit": (
            await asyncio.to_thread(
                subprocess.check_output, ["git", "rev-parse", "HEAD"], text=True
            )
        ).strip(),
        "lock_sha256": initial_lock_hash,
        "source_sha256": initial_source_hashes,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "python": platform.python_version(),
        "client_placement": "separate process, same host, loopback",
        "torch_threads": 2,
        "torch_interop_threads": 1,
        "device": "cpu",
        "precision": "float32",
        "artifact_manifest": json.loads(manifest.read_text())
        if args.adapter != "fake" and await asyncio.to_thread(manifest.exists)
        else None,
        "text_characters": [len(t) for t in TEXTS],
        "texts": TEXTS,
        "warmup_requests": 2 * args.batch_size,
        "client_lag_validity_p99_seconds": 0.02,
        "client_late_drop_seconds": 0.1,
        "client_timeout_seconds": 5.0,
        "server_deadline_seconds": 2.0,
        "response_cache": False,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(
        output,
        {
            "config": vars(args) | {"output": str(output)},
            "provenance": provenance,
            "results": results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", choices=("fake", "custom", "huggingface"), required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("single", "immediate", "timed"),
        default=["single", "immediate", "timed"],
    )
    parser.add_argument("--rates", nargs="+", type=float, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--bursts", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-ms", type=int, default=5)
    parser.add_argument("--pending-capacity", type=int, default=32)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 1 or args.duration <= 0 or not args.rates or min(args.rates) <= 0:
        parser.error("Positive repetitions, duration, and rates required")
    asyncio.run(experiment(args))


if __name__ == "__main__":
    main()
