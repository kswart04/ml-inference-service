from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass

from inference_service.adapters.protocol import ModelAdapter
from inference_service.core.contracts import Prediction, TextInput
from inference_service.core.exceptions import (
    AdapterContractError,
    BatchExecutionError,
    FatalWorkerError,
    QueueFullError,
    RequestDeadlineError,
    SchedulerUnavailableError,
)
from inference_service.core.lifecycle import RequestEnvelope, RequestState
from inference_service.observability.metrics import ServiceMetrics
from inference_service.runtime.config import SchedulingPolicy


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    policy: SchedulingPolicy
    max_batch_size: int
    collection_delay_seconds: float
    pending_capacity: int
    deadline_seconds: float
    shutdown_grace_seconds: float
    watchdog_seconds: float


class ModelScheduler:
    """One event-loop-owned queue and one bounded execution slot for an adapter."""

    def __init__(
        self,
        adapter: ModelAdapter,
        config: SchedulerConfig,
        metrics: ServiceMetrics,
        logger: logging.Logger,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if config.max_batch_size > adapter.metadata.max_batch_size:
            raise ValueError("Configured batch size exceeds adapter maximum.")
        self.adapter = adapter
        self.config = config
        self.metrics = metrics
        self.logger = logger
        self._clock = clock
        self._pending: deque[RequestEnvelope] = deque()
        self._condition = asyncio.Condition()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-worker")
        self._driver: asyncio.Task[None] | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._accepting = False
        self._failed = False
        self._active_batch = False
        self._active_since: float | None = None
        self._running: list[RequestEnvelope] = []
        self._abandoned_worker = False

    def _now(self) -> float:
        if self._clock is not None:
            return self._clock()
        return time.monotonic()

    @property
    def labels(self) -> tuple[str, str, str]:
        key = self.adapter.metadata.key
        return key.model_id, key.model_version, self.config.policy.value

    @property
    def ready(self) -> bool:
        return (
            self._accepting
            and not self._failed
            and self._driver is not None
            and not self._driver.done()
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def active_batch(self) -> bool:
        return self._active_batch

    async def start(self) -> None:
        if self._driver is not None:
            raise RuntimeError("Scheduler already started.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self.adapter.load)
        self._accepting = True
        self._driver = asyncio.create_task(self._run(), name="batch-scheduler")
        self._watchdog = asyncio.create_task(self._watch_worker(), name="worker-watchdog")

    async def submit(self, request_id: str, item: TextInput, *, started_at: float) -> Prediction:
        loop = asyncio.get_running_loop()
        async with self._condition:
            now = self._now()
            self._reclaim_dead_pending(now)
            if now >= started_at + self.config.deadline_seconds:
                self.metrics.requests.labels(*self.labels, "deadline_before_admission").inc()
                raise RequestDeadlineError
            if not self.ready:
                self.metrics.requests.labels(*self.labels, "unavailable").inc()
                raise SchedulerUnavailableError
            if len(self._pending) >= self.config.pending_capacity:
                self.metrics.requests.labels(*self.labels, "queue_full").inc()
                raise QueueFullError
            envelope = RequestEnvelope(
                request_id=request_id,
                item=item,
                request_started_at=started_at,
                admitted_at=now,
                deadline_at=started_at + self.config.deadline_seconds,
                future=loop.create_future(),
            )
            self._pending.append(envelope)
            self.metrics.requests.labels(*self.labels, "accepted").inc()
            self.metrics.pending.labels(*self.labels).set(len(self._pending))
            self._condition.notify_all()

        remaining = envelope.deadline_at - self._now()
        try:
            if remaining <= 0:
                raise TimeoutError
            return await asyncio.wait_for(asyncio.shield(envelope.future), timeout=remaining)
        except TimeoutError:
            await self._terminate(envelope, RequestState.EXPIRED)
            raise RequestDeadlineError from None
        except asyncio.CancelledError:
            await self._terminate(envelope, RequestState.CANCELLED)
            raise

    async def _terminate(self, envelope: RequestEnvelope, state: RequestState) -> None:
        async with self._condition:
            if envelope.terminal:
                return
            envelope.state = state
            if not envelope.future.done():
                envelope.future.cancel()
            self._reclaim_dead_pending(self._now())
            self._record_outcome(envelope, state)
            self._condition.notify_all()

    def _record_outcome(self, envelope: RequestEnvelope, state: RequestState) -> None:
        self.metrics.outcomes.labels(*self.labels, state.value).inc()
        self.metrics.request_duration.labels(*self.labels, state.value).observe(
            max(0.0, self._now() - envelope.request_started_at)
        )
        self.logger.info(
            "request_terminal",
            extra={
                "request_id": envelope.request_id,
                "model_id": self.labels[0],
                "model_version": self.labels[1],
                "policy": self.labels[2],
                "outcome": state.value,
            },
        )

    def _reclaim_dead_pending(self, now: float) -> None:
        retained: deque[RequestEnvelope] = deque()
        while self._pending:
            item = self._pending.popleft()
            if item.state is RequestState.PENDING and now >= item.deadline_at:
                item.state = RequestState.EXPIRED
                if not item.future.done():
                    item.future.cancel()
                self._record_outcome(item, RequestState.EXPIRED)
            if item.state is RequestState.PENDING:
                retained.append(item)
        self._pending = retained
        self.metrics.pending.labels(*self.labels).set(len(self._pending))

    def _take_batch(self, now: float) -> list[RequestEnvelope]:
        self._reclaim_dead_pending(now)
        size = 1 if self.config.policy is SchedulingPolicy.SINGLE else self.config.max_batch_size
        batch: list[RequestEnvelope] = []
        while self._pending and len(batch) < size:
            envelope = self._pending.popleft()
            if envelope.state is RequestState.PENDING:
                envelope.state = RequestState.RUNNING
                batch.append(envelope)
        self.metrics.pending.labels(*self.labels).set(len(self._pending))
        return batch

    def _should_dispatch(self, now: float) -> bool:
        if not self._pending:
            return False
        if self.config.policy is not SchedulingPolicy.TIMED:
            return True
        return len(self._pending) >= self.config.max_batch_size or (
            now - self._pending[0].admitted_at >= self.config.collection_delay_seconds
        )

    async def _next_batch(self) -> list[RequestEnvelope] | None:
        async with self._condition:
            while True:
                now = self._now()
                self._reclaim_dead_pending(now)
                if self._should_dispatch(now):
                    return self._take_batch(now)
                if not self._accepting and not self._pending:
                    return None
                timeout: float | None = None
                if self._pending and self.config.policy is SchedulingPolicy.TIMED:
                    timeout = max(
                        0.0,
                        self._pending[0].admitted_at + self.config.collection_delay_seconds - now,
                    )
                try:
                    if timeout is None:
                        await self._condition.wait()
                    else:
                        await asyncio.wait_for(self._condition.wait(), timeout)
                except TimeoutError:
                    pass

    async def _run(self) -> None:
        try:
            while True:
                batch = await self._next_batch()
                if batch is None:
                    return
                await self._execute(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            self._failed = True
            self._accepting = False
            self.logger.exception("scheduler_failed", extra={"error_category": "scheduler"})
            await self._fail_all(SchedulerUnavailableError(), RequestState.FAILED)

    async def _execute(self, batch: Sequence[RequestEnvelope]) -> None:
        started = self._now()
        for envelope in batch:
            self.metrics.queue_wait.labels(*self.labels).observe(
                max(0.0, started - envelope.admitted_at)
            )
        self.metrics.batch_size.labels(*self.labels).observe(len(batch))
        self.metrics.active_batches.labels(*self.labels).set(1)
        self._active_batch = True
        self._active_since = started
        self._running = list(batch)
        try:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                self._executor, self.adapter.predict_batch, [item.item for item in batch]
            )
            self.metrics.batch_execution.labels(*self.labels).observe(
                max(0.0, self._now() - started)
            )
            timings = self.adapter.last_timings
            self.metrics.preprocessing.labels(*self.labels).observe(timings.preprocessing_seconds)
            self.metrics.forward.labels(*self.labels).observe(timings.forward_seconds)
            self.metrics.postprocessing.labels(*self.labels).observe(timings.postprocessing_seconds)
            if len(results) != len(batch):
                self.metrics.failures.labels(*self.labels, "result_count").inc()
                await self._finish_batch(batch, error=AdapterContractError())
            else:
                await self._finish_batch(batch, results=results)
        except FatalWorkerError:
            self._failed = True
            self._accepting = False
            self.metrics.failures.labels(*self.labels, "fatal_worker").inc()
            await self._finish_batch(batch, error=SchedulerUnavailableError())
            await self._fail_all(SchedulerUnavailableError(), RequestState.FAILED)
            self.logger.error("worker_failed", extra={"error_category": "fatal_worker"})
        except Exception:
            self.metrics.failures.labels(*self.labels, "execution").inc()
            await self._finish_batch(batch, error=BatchExecutionError())
        finally:
            self._active_batch = False
            self._active_since = None
            self._running = []
            self.metrics.active_batches.labels(*self.labels).set(0)

    async def _finish_batch(
        self,
        batch: Sequence[RequestEnvelope],
        *,
        results: Sequence[Prediction] | None = None,
        error: Exception | None = None,
    ) -> None:
        for index, envelope in enumerate(batch):
            if envelope.state is not RequestState.RUNNING:
                continue
            if self._now() >= envelope.deadline_at:
                envelope.state = RequestState.EXPIRED
                if not envelope.future.done():
                    envelope.future.cancel()
                self._record_outcome(envelope, RequestState.EXPIRED)
            elif error is not None:
                envelope.state = RequestState.FAILED
                if not envelope.future.done():
                    envelope.future.set_exception(error)
                self._record_outcome(envelope, RequestState.FAILED)
            else:
                envelope.state = RequestState.SUCCEEDED
                if not envelope.future.done() and results is not None:
                    envelope.future.set_result(results[index])
                self._record_outcome(envelope, RequestState.SUCCEEDED)

    async def _watch_worker(self) -> None:
        interval = min(0.1, self.config.watchdog_seconds / 2)
        try:
            while True:
                await asyncio.sleep(interval)
                if (
                    self._active_since is not None
                    and self._now() - self._active_since >= self.config.watchdog_seconds
                ):
                    self._failed = True
                    self._accepting = False
                    self.logger.error("worker_stuck", extra={"error_category": "watchdog"})
        except asyncio.CancelledError:
            raise

    async def _fail_all(self, error: Exception, state: RequestState) -> None:
        async with self._condition:
            pending = [*self._pending, *self._running]
            self._pending.clear()
            self.metrics.pending.labels(*self.labels).set(0)
            for envelope in pending:
                if not envelope.terminal:
                    envelope.state = state
                    if not envelope.future.done():
                        envelope.future.set_exception(error)
                    self._record_outcome(envelope, state)
            self._condition.notify_all()

    async def shutdown(self) -> None:
        self._accepting = False
        async with self._condition:
            self._condition.notify_all()
        if self._driver is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._driver), self.config.shutdown_grace_seconds
                )
            except TimeoutError:
                await self._fail_all(SchedulerUnavailableError(), RequestState.FAILED)
                self._abandoned_worker = self._active_batch
                self._driver.cancel()
                with suppress(asyncio.CancelledError):
                    await self._driver
        if self._watchdog is not None:
            self._watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await self._watchdog
        if not self._abandoned_worker:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self.adapter.close)
        # An abandoned native call cannot be killed. Avoid blocking event-loop shutdown.
        self._executor.shutdown(wait=False, cancel_futures=True)
