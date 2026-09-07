# Decision log

## D001 — Incremental scope and private repository (2026-09-06)

**Decision:** use the owner's `kswart04/ml-inference-service` private repository
and complete M0 before beginning M1. Keep a section-by-section work log, architecture
notes, and learning notes alongside the code.

**Reason:** the requirements prioritize explainable engineering and verified
milestones. A single large implementation would obscure correctness gates.

**Consequence:** no real model downloads, training, benchmark claims, hosted
deployment, or public release in M0. Code license remains undecided. Git commit
identity is configured only for this repository with GitHub's no-reply address.

## D002 — Python 3.12 and locked uv environment (2026-09-06)

**Decision:** target Python 3.12 only for the initial baseline; commit
`.python-version`, `pyproject.toml`, and `uv.lock`. Use Hatchling for the src package
build and uv for environment management. Add ML dependencies in their own milestones.

**Reason:** one explicit interpreter series keeps initial verification focused;
the machine's default Python 3.14 is not automatically the project baseline.
The fake adapter requires no PyTorch or model downloads.

**Consequence:** supporting additional Python versions needs explicit testing.
This choice does not establish compatibility of future PyTorch/CUDA dependency
sets; verify and lock those during M2/M3. Normal setup uses `uv sync --locked` so
manifest/lock drift fails visibly.

Resolved M0 direct dependencies: FastAPI 0.141.1, Pydantic 2.13.5,
pydantic-settings 2.15.0, Starlette 1.6.0, Uvicorn 0.52.4. Development tools:
HTTPX 0.28.1, mypy 1.20.2, pytest 9.1.1, pytest-asyncio 1.4.0, Ruff 0.16.6.
The lockfile is authoritative for exact versions and hashes.

References: [uv project files](https://docs.astral.sh/uv/concepts/projects/layout/),
[uv Python management](https://docs.astral.sh/uv/guides/install-python/).

## D003 — Typed adapters and fixed explicit model versions (2026-09-06)

**Decision:** separate HTTP schemas from the adapter protocol; share typed inputs,
predictions, and immutable identity structures. Use a fixed startup registry.

**Reason:** scheduling must not need to know the model architecture. A stable
version travels from lookup to response and will identify compatible batches.

**Consequence:** v1 and v2 cannot accidentally resolve to the same key. The current
sentiment schema intentionally does not claim support for other tasks. If future
request options affect compatibility, extend the key before accepting those options.

## D004 — Minimal fake execution before scheduling (2026-09-06)

**Decision:** invoke only the cheap fake adapter inline in M0, with exactly one
input; use FastAPI lifespan for load and close. Implement bounded worker execution
and all three scheduling policies together in M1.

**Reason:** M0 establishes interfaces and an API without implicitly introducing an
executor queue or prematurely claiming scheduler correctness.

**Consequence:** M0 is a local scaffold, unsuitable for real/slow inference. The
single input path is not the implemented M1 single-item scheduling policy. Load,
preprocessing, and model forward passes must move off the event loop before real
models are introduced.

Reference: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
For M1's shutdown design, Python documents that pending thread-pool work can keep
the interpreter alive even when shutdown does not wait:
[executor shutdown](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.shutdown).

## D005 — Validate at the transport and adapter boundaries (2026-09-06)

**Decision:** enforce a byte cap before parsing, typed/strict request schemas, and
a configurable character cap before invoking the adapter. Use consistent safe
error envelopes and server-generated IDs.

**Reason:** byte size and character count bound different work. IDs identify one
prediction attempt and must not imply retry deduplication.

**Consequence:** error details deliberately omit raw submitted text and tracebacks.
Token limits remain a separate real-adapter requirement for M2/M3. M1 adds
structured logging, deadline timing, and per-model pending-capacity bounds.

## D006 — One scheduler driver and one execution thread (2026-09-06)

**Decision:** create one `ModelScheduler` per configured model identity. One event-loop
driver owns batch selection and is the only code that submits to a dedicated
single-thread executor. It awaits the active batch before submitting another.

**Reason:** this makes both pending capacity and running capacity visible. A normal
executor submission queue cannot silently become a second unbounded work queue.

**Consequence:** M1 permits one active batch per scheduler. Concurrent models on a
shared GPU remain deferred because per-model limits do not bound device-wide work.

## D007 — Terminal state ownership remains on the event loop (2026-09-06)

**Decision:** represent accepted requests as pending, running, succeeded, failed,
expired, or cancelled. Worker threads return ordered values or errors; only event-loop
code mutates envelopes and futures.

**Reason:** one owner makes exactly-once terminal resolution and result correlation
reviewable under timeout, disconnect, and batch failure races.

**Consequence:** cancelling a running waiter never releases the execution slot.
Late output is ignored. A thread blocked in native code cannot be terminated; the
watchdog changes readiness and process restart remains the recovery mechanism.

## D008 — Per-app metrics and fixed structured-log fields (2026-09-06)

**Decision:** use a separate Prometheus registry for each app instance and JSON logs
with a fixed field allowlist. Labels use configured identity, policy, and small
decision/outcome categories.

**Reason:** isolated registries make repeated app construction safe in tests and
avoid global duplicate collectors. Fixed fields keep request IDs and raw text out
of metric labels and keep text out of normal logs.

**Consequence:** preprocessing and postprocessing collectors are declared but remain
unobserved for the M1 fake. M2 adapters must expose reliable phase timings before
those series carry samples.
