import asyncio

import httpx
import pytest

from benchmarks.load import arrivals, metric_values, percentile, run_load


def test_absolute_arrivals_do_not_depend_on_response_times() -> None:
    assert arrivals(4, 1) == [0, 0.25, 0.5, 0.75]
    burst = arrivals(100, 1, True)
    assert all(t % 0.5 < 0.1 for t in burst)
    assert len(burst) == 100
    with pytest.raises(ValueError):
        arrivals(0, 1)


def test_quantiles_and_metric_labels_preserve_errors() -> None:
    assert percentile([], 0.99) is None
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    metrics = metric_values(
        "# TYPE inference_terminal_outcomes_total counter\n"
        'inference_terminal_outcomes_total{model_id="m",policy="single",outcome="expired"} 3\n'
    )
    assert metrics["inference_terminal_outcomes_total{outcome=expired}"] == 3


async def test_open_loop_records_errors_and_bounded_client_drops() -> None:
    active = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.03)
        active -= 1
        return httpx.Response(429, json={"error": "overload"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    ) as client:
        result = await run_load(
            client,
            {"model_id": "m", "model_version": "v"},
            rate=1000,
            duration=0.02,
            max_outstanding=1,
        )
    assert peak == 1
    assert result["intended"] == sum(result["outcomes"].values())
    assert result["sent"] == result["outcomes"]["429"]
    assert result["client_drops"] > 0
    assert not result["client_valid"]
    assert result["final_outstanding"] == 0
    assert result["success_latency_seconds"]["p95"] is None
    assert result["outcome_duration_seconds"]["429"]["p95"] > 0
