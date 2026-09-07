"""Bounded open-loop HTTP load with explicit intended arrivals and client drops."""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

TEXTS = (
    "Excellent.",
    "This was disappointing and I would not recommend it.",
    "The acting was convincing, although the story took a long time to get going.",
    "The service was friendly and the food was delicious. " * 4,
    "The product broke after a week and customer support could not help. " * 12,
    "I enjoyed parts of the film but the ending was predictable. " * 24,
)


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def arrivals(rate: float, duration: float, burst: bool = False) -> list[float]:
    if rate <= 0 or duration <= 0 or rate * duration > 1_000_000:
        raise ValueError("Rate/duration must be positive and at most one million arrivals")
    steady = [i / rate for i in range(math.ceil(rate * duration)) if i / rate < duration]
    # Compress each half-second's arrivals into its first 100 ms; preserve exact count.
    return [math.floor(t / 0.5) * 0.5 + (t % 0.5) / 5 for t in steady] if burst else steady


def metric_values(payload: str) -> dict[str, float]:
    result = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            if not sample.name.startswith("inference_") or sample.name.endswith("_created"):
                continue
            labels = ",".join(
                f"{key}={value}"
                for key, value in sorted(sample.labels.items())
                if key not in {"model_id", "model_version", "policy"}
            )
            result[sample.name + ("{" + labels + "}" if labels else "")] = float(sample.value)
    return result


async def run_load(
    client: httpx.AsyncClient,
    identity: dict[str, str],
    *,
    rate: float,
    duration: float,
    burst: bool = False,
    max_outstanding: int = 256,
    lag_limit: float = 0.1,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    if max_outstanding <= 0 or lag_limit <= 0:
        raise ValueError("Outstanding and lag limits must be positive")
    schedule = arrivals(rate, duration, burst)
    before = metric_values((await client.get("/metrics")).text)
    rows: list[dict[str, Any]] = []
    tasks: set[asyncio.Task[None]] = set()
    peak = 0
    start = time.perf_counter()

    async def send(index: int, intended: float) -> None:
        sent = time.perf_counter() - start
        outcome = "transport_error"
        try:
            response = await client.post(
                "/v1/predict", json={**identity, "input": {"text": TEXTS[index % len(TEXTS)]}}
            )
            outcome = str(response.status_code)
            if response.status_code == 200:
                body = response.json()
                if (
                    body.get("model_id") != identity["model_id"]
                    or body.get("model_version") != identity["model_version"]
                    or "prediction" not in body
                ):
                    outcome = "invalid_response"
        except httpx.TimeoutException:
            outcome = "client_timeout"
        except (httpx.HTTPError, ValueError):
            outcome = "transport_error"
        end = time.perf_counter() - start
        rows.append(
            {
                "index": index,
                "intended_s": intended,
                "sent_s": sent,
                "lag_s": sent - intended,
                "duration_s": end - sent,
                "completed_s": end,
                "outcome": outcome,
            }
        )

    for index, intended in enumerate(schedule):
        remaining = start + intended - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        lag = time.perf_counter() - start - intended
        reason = (
            "client_late_drop"
            if lag > lag_limit
            else ("client_capacity_drop" if len(tasks) >= max_outstanding else None)
        )
        if reason:
            rows.append(
                {
                    "index": index,
                    "intended_s": intended,
                    "sent_s": None,
                    "lag_s": lag,
                    "duration_s": None,
                    "completed_s": None,
                    "outcome": reason,
                }
            )
            continue
        task = asyncio.create_task(send(index, intended))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        peak = max(peak, len(tasks))
    await asyncio.sleep(max(0.0, start + duration - time.perf_counter()))
    outstanding_at_end = len(tasks)
    if tasks:
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    after = metric_values((await client.get("/metrics")).text)
    # Timed-out native work can outlive its HTTP waiter. Do not contaminate the next run.
    recovery_started = time.perf_counter()
    while after.get("inference_pending_requests", 0.0) or after.get(
        "inference_active_batches", 0.0
    ):
        if time.perf_counter() - recovery_started > 30:
            raise RuntimeError("Server did not drain after measurement")
        await asyncio.sleep(0.05)
        after = metric_values((await client.get("/metrics")).text)
    recovery_seconds = time.perf_counter() - recovery_started
    outcomes = dict(Counter(row["outcome"] for row in rows))
    successful = [r["duration_s"] for r in rows if r["outcome"] == "200"]
    drops = sum(count for name, count in outcomes.items() if name.endswith("_drop"))
    lag_distribution = distribution([r["lag_s"] for r in rows])
    sent_count = sum(r["sent_s"] is not None for r in rows)
    completed_in_window = sum(r["outcome"] == "200" and r["completed_s"] <= duration for r in rows)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw_path, "wt") as handle:
            for row in sorted(rows, key=lambda r: r["index"]):
                handle.write(json.dumps(row) + "\n")
    return {
        "rate_requested": rate,
        "burst": burst,
        "scheduled_seconds": duration,
        "elapsed_with_drain_seconds": elapsed,
        "server_recovery_seconds": recovery_seconds,
        "intended": len(schedule),
        "sent": sent_count,
        "outcomes": outcomes,
        "client_drops": drops,
        "successful_rps_in_window": completed_in_window / duration,
        "successful_rps_with_drain": len(successful) / elapsed,
        "success_latency_seconds": distribution(successful),
        "scheduling_lag_seconds": lag_distribution,
        "outcome_duration_seconds": {
            name: distribution(
                [
                    r["duration_s"]
                    for r in rows
                    if r["outcome"] == name and r["duration_s"] is not None
                ]
            )
            for name in outcomes
        },
        "outstanding_at_window_end": outstanding_at_end,
        "final_outstanding": len(tasks),
        "peak_outstanding": peak,
        "max_outstanding": max_outstanding,
        "client_valid": drops == 0 and (lag_distribution["p99"] or 0) <= 0.02,
        "metrics_delta": {name: value - before.get(name, 0.0) for name, value in after.items()},
        "final_pending": after.get("inference_pending_requests", 0.0),
        "final_active_batches": after.get("inference_active_batches", 0.0),
    }
