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

**Consequence:** the M1 fake records zero-valued phases. The M2 adapter supplies
preprocessing, synchronized forward, and postprocessing durations through the same
protocol for every policy.

## D009 — Pinned local Hugging Face artifacts (2026-09-07)

**Decision:** support one allowlisted Hub repository at immutable commit
`714eb0fa89d2f80546fda750413ed43d93601a13`. A separate preparation script downloads
five required files and writes their SHA-256 manifest. Runtime loads only the local
directory with remote code, network lookup, and non-safetensors weights disabled.

**Reason:** model acquisition is an operator action with visible provenance. It
cannot be smuggled into an untrusted prediction request or silently change when a
branch advances.

**Consequence:** a fresh checkout must run preparation before selecting the adapter.
The roughly 256 MiB artifact remains outside Git. Any file or preprocessing change
requires a new internal version and manifest rather than reusing the current ID.

## D010 — Longest padding, 256-token truncation, verified labels (2026-09-07)

**Decision:** tokenize one whole batch with longest-item padding, attention masks,
and truncation at 256 tokens. Require upstream IDs 0/1 to map to negative/positive.

**Reason:** dynamic shapes reduce padding relative to always padding to 256, while
the fixed ceiling bounds tensor size. Startup validation prevents silent label inversion.

**Consequence:** prediction can truncate text even when it is below the separate
8,000-character admission limit. The max length is part of internal model version
`hf-sst2-714eb0fa-max256`.

## D011 — Optional ML dependencies and explicit model tests (2026-09-07)

**Decision:** keep PyTorch, Transformers, and Hugging Face Hub in the `hf` optional
extra. The default pytest gate excludes tests marked `model`; the explicit M2 gate
runs them offline after artifact preparation.

**Reason:** scheduler development and CI can remain fast and offline without hiding
whether the real model was tested. Missing artifacts produce a setup-oriented skip.

**Consequence:** use `uv sync --locked --extra hf` and `pytest -m model` for M2.
The verified CPU lock resolves PyTorch 2.14.0, Transformers 5.16.1, and Hub 1.30.0.

## D012 — UCI sentences and a small CPU baseline (2026-09-07)

**Decision:** use the official UCI Sentiment Labelled Sentences archive, pinned by
SHA-256, with its explicit CC BY 4.0 attribution. Use the standard library for this
82 KB archive instead of adding a dataset framework.

**Reason:** the requirements allow a public sentiment dataset such as IMDb. This
source has clear licensing metadata and a bounded three-domain sentiment task.
The proposed Python/FastAPI/PyTorch stack remains unchanged.

**Consequence:** results describe this small sentence benchmark, not performance on
the full IMDb review dataset. Near duplicates and related parent reviews may remain
despite removing exact normalized-text overlap before splitting.

## D013 — Separate fitting, selection, and final evaluation (2026-09-07)

**Decision:** split deterministically before vocabulary fitting, fit on selected
training rows only, select the checkpoint on validation macro-F1, and evaluate an
immutable export with a separate test command. Provide small and full CPU profiles.

**Reason:** test-based vocabulary fitting or checkpoint selection would make the
held-out quality claim unreliable. Separate commands make the data boundary reviewable.

**Consequence:** output overwrite is refused. The reported test measurement is final
for this configuration; it must not become a tuning target. A seven-point gap to
validation is documented, not hidden by retraining after seeing test results.

## D014 — Shared model code and content-checked exports (2026-09-07)

**Decision:** share token encoding and the masked-mean PyTorch network between
training and serving. Export safetensors, vocabulary, typed config, and provenance;
derive the model version from the manifest's file hashes.

**Reason:** training/serving preprocessing drift would invalidate reload evidence.
Masking the pooling denominator is essential for predictions to remain stable when
neighbors have different lengths. A strict manifest catches changed artifacts.

**Consequence:** an independent process can reconstruct the model offline. The
provenance report includes elapsed time, so a deterministic retraining can produce
the same weights but a different version. Integrity hashes detect changes; they do
not establish authenticity of arbitrary untrusted exports. Paths remain operator-owned.
