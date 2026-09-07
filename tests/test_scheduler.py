from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import replace

import pytest
from starlette.requests import Request
from starlette.types import Message

from inference_service.adapters.fake import FakeAdapter
from inference_service.api.app import _await_prediction
from inference_service.api.schemas import PredictRequest
from inference_service.core.contracts import ModelKey, ModelMetadata, Prediction, TextInput
from inference_service.core.exceptions import (
    AdapterContractError,
    BatchExecutionError,
    FatalWorkerError,
    QueueFullError,
    RequestDeadlineError,
    SchedulerUnavailableError,
)
from inference_service.core.scheduler import ModelScheduler, SchedulerConfig
from inference_service.observability.metrics import ServiceMetrics
from inference_service.runtime.config import SchedulingPolicy


class ControlledAdapter(FakeAdapter):
    def __init__(self, *, version: str = "v1") -> None:
        super().__init__()
        self.version = version
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls: list[list[str]] = []
        self.fail = False
        self.wrong_count = False
        self.fatal = False

    @property
    def metadata(self) -> ModelMetadata:
        return replace(super().metadata, key=ModelKey("fake-sentiment", self.version))

    def predict_batch(self, items: Sequence[TextInput]) -> list[Prediction]:
        self.calls.append([item.text for item in items])
        self.started.set()
        assert self.release.wait(timeout=3), "test did not release controlled worker"
        if self.fail:
            raise RuntimeError("controlled failure")
        if self.fatal:
            raise FatalWorkerError("controlled fatal worker failure")
        results = super().predict_batch(items)
        return results[:-1] if self.wrong_count else results


def config(
    *,
    policy: SchedulingPolicy = SchedulingPolicy.TIMED,
    batch_size: int = 4,
    delay: float = 0.05,
    capacity: int = 8,
    deadline: float = 1.0,
    grace: float = 0.2,
    watchdog: float = 1.0,
) -> SchedulerConfig:
    return SchedulerConfig(policy, batch_size, delay, capacity, deadline, grace, watchdog)


def make_scheduler(adapter: ControlledAdapter, cfg: SchedulerConfig) -> ModelScheduler:
    return ModelScheduler(adapter, cfg, ServiceMetrics(), logging.getLogger("test.scheduler"))


async def submit(
    scheduler: ModelScheduler,
    number: int,
    text: str | None = None,
) -> Prediction:
    return await scheduler.submit(
        f"request-{number}",
        TextInput(text=text or f"good {number}"),
        started_at=time.monotonic(),
    )


async def wait_thread(event: threading.Event, wait_seconds: float = 1.0) -> None:
    assert await asyncio.to_thread(event.wait, wait_seconds)


@pytest.mark.realtime
async def test_t01_results_stay_correlated_when_one_running_client_times_out() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(batch_size=3, delay=1, deadline=0.12))
    await scheduler.start()
    tasks = [asyncio.create_task(submit(scheduler, 0, "good"))]
    await asyncio.sleep(0.08)
    tasks.extend(
        asyncio.create_task(submit(scheduler, i, text))
        for i, text in enumerate(("bad", "great"), start=1)
    )
    await wait_thread(adapter.started)
    await asyncio.sleep(0.06)
    adapter.release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert isinstance(results[0], RequestDeadlineError)
    assert [result.label for result in results[1:] if isinstance(result, Prediction)] == [
        "negative",
        "positive",
    ]
    await scheduler.shutdown()


async def test_pending_expiry_before_waiter_timer_is_a_deadline_error() -> None:
    now = [0.0]
    scheduler = ModelScheduler(
        ControlledAdapter(),
        config(delay=10, capacity=1),
        ServiceMetrics(),
        logging.getLogger("test.expiry"),
        clock=lambda: now[0],
    )
    await scheduler.start()
    first = asyncio.create_task(scheduler.submit("first", TextInput(text="good"), started_at=0))
    await asyncio.sleep(0)
    now[0] = 2.0
    second = asyncio.create_task(scheduler.submit("second", TextInput(text="good"), started_at=2))
    try:
        with pytest.raises(RequestDeadlineError):
            await first
        assert scheduler.pending_count == 1
    finally:
        second.cancel()
        await asyncio.gather(second, return_exceptions=True)
        await scheduler.shutdown()


async def test_finished_batch_after_deadline_returns_deadline_error() -> None:
    now = [0.0]
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = ModelScheduler(
        adapter,
        config(delay=0),
        ServiceMetrics(),
        logging.getLogger("test.expiry"),
        clock=lambda: now[0],
    )
    await scheduler.start()
    task = asyncio.create_task(scheduler.submit("running", TextInput(text="good"), started_at=0))
    try:
        await wait_thread(adapter.started)
        now[0] = 2.0
        adapter.release.set()
        with pytest.raises(RequestDeadlineError):
            await task
        assert not scheduler.active_batch
    finally:
        adapter.release.set()
        await scheduler.shutdown()


async def test_http_handler_cancellation_joins_children_and_reclaims_pending() -> None:
    scheduler = make_scheduler(ControlledAdapter(), config(delay=10))
    await scheduler.start()
    received: asyncio.Queue[Message] = asyncio.Queue()
    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/predict"}, receive=received.get
    )
    request.state.request_id = "cancel-parent"
    request.state.started_at = time.monotonic()
    existing_tasks = asyncio.all_tasks()
    handler = asyncio.create_task(
        _await_prediction(
            scheduler,
            request,
            PredictRequest(
                model_id="fake-sentiment", model_version="v1", input=TextInput(text="good")
            ),
        )
    )
    try:
        async with scheduler._condition:
            await asyncio.wait_for(
                scheduler._condition.wait_for(lambda: scheduler.pending_count == 1), timeout=1
            )
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(handler, timeout=0.5)
        assert scheduler.pending_count == 0
        assert not (asyncio.all_tasks() - existing_tasks)
    finally:
        await scheduler.shutdown()


async def test_t02_full_batch_dispatches_before_long_window() -> None:
    adapter = ControlledAdapter()
    scheduler = make_scheduler(adapter, config(batch_size=3, delay=10))
    await scheduler.start()
    tasks = [asyncio.create_task(submit(scheduler, i)) for i in range(3)]
    await wait_thread(adapter.started, 0.5)
    assert len(adapter.calls[0]) == 3
    await asyncio.gather(*tasks)
    await scheduler.shutdown()


@pytest.mark.realtime
async def test_t03_partial_batch_dispatches_after_oldest_window() -> None:
    adapter = ControlledAdapter()
    scheduler = make_scheduler(adapter, config(delay=0.04))
    await scheduler.start()
    started = time.monotonic()
    result = await submit(scheduler, 1)
    elapsed = time.monotonic() - started
    assert result.label == "positive"
    assert 0.025 <= elapsed < 0.5
    assert adapter.calls == [["good 1"]]
    await scheduler.shutdown()


@pytest.mark.realtime
async def test_t04_busy_worker_does_not_restart_collection_window() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(batch_size=2, delay=0.08))
    await scheduler.start()
    first = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    second = asyncio.create_task(submit(scheduler, 2))
    await asyncio.sleep(0.1)
    released = time.monotonic()
    adapter.release.set()
    await asyncio.gather(first, second)
    assert time.monotonic() - released < 0.06
    assert adapter.calls == [["good 1"], ["good 2"]]
    await scheduler.shutdown()


async def test_t05_atomic_admission_never_exceeds_pending_capacity() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(policy=SchedulingPolicy.SINGLE, capacity=1))
    await scheduler.start()
    running = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    pending = asyncio.create_task(submit(scheduler, 2))
    await asyncio.sleep(0)
    rejected = await asyncio.gather(
        submit(scheduler, 3), submit(scheduler, 4), return_exceptions=True
    )
    assert all(isinstance(item, QueueFullError) for item in rejected)
    assert scheduler.pending_count == 1
    adapter.release.set()
    await asyncio.gather(running, pending)
    await scheduler.shutdown()


async def test_t06_cancelled_pending_work_reclaims_capacity_and_never_executes() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(policy=SchedulingPolicy.SINGLE, capacity=1))
    await scheduler.start()
    running = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    abandoned = asyncio.create_task(submit(scheduler, 2))
    await asyncio.sleep(0)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    replacement = asyncio.create_task(submit(scheduler, 3))
    adapter.release.set()
    await asyncio.gather(running, replacement)
    assert [item for call in adapter.calls for item in call] == ["good 1", "good 3"]
    await scheduler.shutdown()


@pytest.mark.realtime
async def test_t07_running_timeout_keeps_slot_and_survivor_runs_afterward() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(policy=SchedulingPolicy.SINGLE, deadline=0.12))
    await scheduler.start()
    expired = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    await asyncio.sleep(0.08)
    survivor = asyncio.create_task(submit(scheduler, 2))
    with pytest.raises(RequestDeadlineError):
        await expired
    assert scheduler.active_batch
    assert len(adapter.calls) == 1
    adapter.release.set()
    assert (await survivor).label == "positive"
    assert adapter.calls == [["good 1"], ["good 2"]]
    await scheduler.shutdown()


async def test_t08_versions_use_isolated_queues_and_batches() -> None:
    first, second = ControlledAdapter(version="v1"), ControlledAdapter(version="v2")
    schedulers = [make_scheduler(first, config(delay=0)), make_scheduler(second, config(delay=0))]
    await asyncio.gather(*(scheduler.start() for scheduler in schedulers))
    await asyncio.gather(submit(schedulers[0], 1), submit(schedulers[1], 2))
    assert first.calls == [["good 1"]]
    assert second.calls == [["good 2"]]
    await asyncio.gather(*(scheduler.shutdown() for scheduler in schedulers))


@pytest.mark.parametrize("wrong_count", [False, True])
async def test_t09_batch_failure_resolves_all_waiters_once(wrong_count: bool) -> None:
    adapter = ControlledAdapter()
    adapter.fail = not wrong_count
    adapter.wrong_count = wrong_count
    scheduler = make_scheduler(adapter, config(batch_size=2, delay=0))
    await scheduler.start()
    results = await asyncio.gather(
        submit(scheduler, 1), submit(scheduler, 2), return_exceptions=True
    )
    expected = AdapterContractError if wrong_count else BatchExecutionError
    assert all(isinstance(result, expected) for result in results)
    assert not scheduler.active_batch
    await scheduler.shutdown()


async def test_t11_shutdown_refuses_new_work_and_bounds_existing_waiter() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(delay=0, grace=0.03, deadline=2))
    await scheduler.start()
    waiter = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    await scheduler.shutdown()
    with pytest.raises(SchedulerUnavailableError):
        await waiter
    with pytest.raises(SchedulerUnavailableError):
        await submit(scheduler, 2)
    adapter.release.set()


async def test_t14_watchdog_marks_stuck_worker_unready() -> None:
    adapter = ControlledAdapter()
    adapter.release.clear()
    scheduler = make_scheduler(adapter, config(delay=0, deadline=2, watchdog=0.03, grace=0.02))
    await scheduler.start()
    waiter = asyncio.create_task(submit(scheduler, 1))
    await wait_thread(adapter.started)
    await asyncio.sleep(0.06)
    assert not scheduler.ready
    await scheduler.shutdown()
    adapter.release.set()
    with pytest.raises(SchedulerUnavailableError):
        await waiter


async def test_t14_fatal_worker_failure_marks_scheduler_unready() -> None:
    adapter = ControlledAdapter()
    adapter.fatal = True
    scheduler = make_scheduler(adapter, config(delay=0))
    await scheduler.start()
    with pytest.raises(SchedulerUnavailableError):
        await submit(scheduler, 1)
    assert not scheduler.ready
    with pytest.raises(SchedulerUnavailableError):
        await submit(scheduler, 2)
    await scheduler.shutdown()


async def test_t15_repeated_success_cancel_timeout_waves_leave_no_queue_growth() -> None:
    adapter = ControlledAdapter()
    scheduler = make_scheduler(adapter, config(delay=0, deadline=0.03))
    await scheduler.start()
    for wave in range(5):
        successful = [asyncio.create_task(submit(scheduler, wave * 10 + i)) for i in range(3)]
        cancelled = asyncio.create_task(submit(scheduler, wave * 10 + 8))
        cancelled.cancel()
        expired = asyncio.create_task(
            scheduler.submit(
                f"expired-{wave}",
                TextInput(text="bad"),
                started_at=time.monotonic() - 1,
            )
        )
        await asyncio.gather(*successful)
        await asyncio.gather(cancelled, return_exceptions=True)
        with pytest.raises(RequestDeadlineError):
            await expired
        assert scheduler.pending_count == 0
        assert not scheduler.active_batch
    await scheduler.shutdown()
