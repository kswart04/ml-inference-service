# Work log

Record each section of work here: purpose, files or tools changed, verification,
and remaining limitations. Keep requirements in `REQUIREMENTS.md` authoritative;
this log records observed progress rather than changing release gates.

## 2026-09-06 — Repository setup

**Purpose:** establish a private GitHub repository before implementing M0.

- Inspected the workspace: only `docs/REQUIREMENTS.md` was present; no local Git
  repository or existing remote was found.
- Checked parent and project locations for `AGENTS.md`; none was found.
- Found Git and Homebrew Python 3.14.7. GitHub CLI, uv, and a configured Git commit
  identity were absent.
- Installed GitHub CLI 2.100.0 and uv 0.12.10 through Homebrew with approval.
- Completed GitHub browser authentication as `kswart04`.
- Added `.gitignore` for environments, secrets, downloaded artifacts, datasets,
  and bulky benchmark output.
- The owner explicitly requested a private GitHub repository. Repository name:
  `ml-inference-service`, subject to checking availability in the signed-in account.
- Code licensing remains an owner decision; no license has been selected.

- Created the private repository at
  <https://github.com/kswart04/ml-inference-service> after checking availability.
- Initialized local Git on `main`. Configured repository-local commit identity
  `kswart04` with GitHub's ID-based no-reply email; global Git identity is unchanged.
- Installed Python 3.12.14 with uv for the project's Python 3.12 baseline.

**Verification:** tool version checks and repository creation succeeded. Initial
commit `95595cd` was pushed to `origin/main`. GitHub reports visibility `PRIVATE`
and default branch `main` for `kswart04/ml-inference-service`.

## 2026-09-06 — M0 package and tooling

**Purpose:** make the project installable and establish baseline checks.

- Added the src-based `inference_service` package, Hatchling build configuration,
  Python 3.12 baseline, uv manifest and lockfile.
- Installed only API/runtime and development dependencies; no ML packages, model
  weights, or datasets were downloaded.
- Established Ruff lint/format, strict mypy, pytest, and pytest-asyncio settings.
- Declared Starlette directly because the API imports its middleware types and
  responses; the resolver selects a version compatible with FastAPI.

**Verification:** `uv sync --locked` succeeds in a fresh temporary copy with its
own virtual environment. Exact resolved dependency versions are in `uv.lock` and
summarized in [D002](DECISIONS.md#d002--python-312-and-locked-uv-environment-2026-09-06).

## 2026-09-06 — M0 contracts and fake adapter

**Purpose:** define the boundary reusable by future schedulers and real models.

- Added typed input, binary prediction, scores, immutable identity and metadata.
- Added the adapter protocol for load, validate, compatibility, batch prediction,
  and close.
- Added a fixed model registry that rejects duplicate identities and supports
  exact version lookup without mutable aliases.
- Added `fake-sentiment/v1`, a deterministic lexical fixture; documented its fixed
  scores, tie behavior, batch size contract, and lack of model-quality meaning.

**Verification:** tests prove ordered/repeated output, fake single/batch parity,
load/close behavior, batch-size rejection, version distinction, and immutable
identity. This is not T12 real-model parity or T08 scheduler isolation completion.

## 2026-09-06 — M0 API, configuration, and lifecycle

**Purpose:** provide a runnable local API with useful validation boundaries.

- Added `POST /v1/predict`, `GET /v1/models`, liveness and readiness endpoints.
- Required explicit model versions and rejected unknown request fields.
- Added byte-counting ASGI middleware before JSON parsing, a character limit before
  prediction, and server-generated UUIDs in response headers and relevant bodies.
- Added safe, consistent validation, lookup, unavailability, and internal-error
  responses; preserved HTTP method error headers.
- Added validated environment settings and FastAPI lifespan loading/cleanup.
- Kept prediction deliberately inline with the cheap fake only; worker execution
  and scheduler lifecycle are M1 work, as explained in D004.

**Verification:** API tests cover blank/invalid text, required fields, unknown
versions, configured character boundaries, malformed JSON, byte-cap boundaries,
chunked UTF-8 input, safe exceptions, wrong output count, request IDs, and readiness
before/during/after lifespan. Real Uvicorn HTTP requests confirmed both health
endpoints, model listing, and the README prediction example return 200. Ctrl-C
produced application shutdown completion and server exit.

## 2026-09-06 — M0 documentation and final checks

**Purpose:** document every section and make the next milestone understandable.

- Added README setup/run/test commands, configuration, exact implemented scope,
  error behavior, and remaining milestones.
- Added architecture and current request-flow documentation.
- Recorded five decisions with reasons and consequences, and learning notes on
  execution ownership, timeouts, correlation, and benchmark interpretation.
- Preserved the supplied requirements document without modifying its gates.

**Verification environment:** macOS on Apple Silicon; uv 0.12.10; Python 3.12.14.
Copied the source, tests, docs, manifest, and lock into a fresh temporary directory
with no preexisting virtual environment. Dependency artifacts could use uv's local
download cache. Ran the documented commands there:

| Command | Result |
| --- | --- |
| `uv sync --locked` | Passed; fresh environment installed |
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | Passed |
| `uv run mypy` | Passed; 17 source files |
| `uv run pytest` | 32 passed; 2 upstream deprecation warnings |
| README Uvicorn command and four curl examples | Passed over localhost HTTP |

The initial type check caught a deliberate frozen-field assignment in a negative
test. Added a narrow type-ignore annotation for that runtime immutability test;
the final strict type check passes.

Known dependency warnings remain visible: Starlette's synchronous TestClient warns
about its HTTPX integration and an AnyIO `BlockingPortal` alias. They do not fail
tests. Revisit the test client integration when developing M1's async suite;
warnings have not been suppressed.

**M0 gate: passed.** The service runs locally, fake prediction and invalid-input
checks pass, and architecture/dependency choices are documented. This is a fresh
environment verification of M0, not the full M4 clean-checkout release gate.

**Next: M1.** Implement bounded worker execution, all three policies, admission,
deadlines, cancellation, shutdown/watchdog behavior, metrics, structured logs, and
the applicable concurrency acceptance tests. Real adapters, training, benchmarks,
Docker, CI, and final release documentation remain incomplete. No throughput or
model-quality claims have been made.

## 2026-09-06 — M1 scheduler and execution ownership

**Purpose:** add dynamic batching without introducing an unbounded executor backlog.

- Added request envelopes and explicit pending/running/terminal lifecycle states.
- Added one per-model FIFO pending deque, atomic admission, three policies, and
  oldest-item timed collection.
- Added one scheduler driver and one dedicated execution thread. The driver awaits
  each active batch before submitting another.
- Moved adapter load, prediction, and ordinary close off the event loop.
- Added ordered result-count validation and event-loop-only future resolution.

**Verification:** scheduler tests T01–T09 prove correlation, full/partial dispatch,
busy-worker window behavior, atomic capacity, pending cleanup, retained execution
slots after running timeout, version isolation, and batch-wide error resolution.

## 2026-09-06 — M1 deadlines, overload, shutdown, and readiness

**Purpose:** give every accepted or rejected request a bounded, explicit outcome.

- Starts deadlines in raw-request middleware before body parsing and uses monotonic
  time for every duration.
- Added 429 overload responses with configurable `Retry-After`, 504 deadlines, and
  503 responses for draining or unavailable workers.
- Added disconnect polling that cancels the scheduler waiter. Pending tombstones are
  reclaimed; running work retains its execution slot.
- Added graceful draining, bounded remaining-waiter failure, fatal-worker signaling,
  and a watchdog that marks an overlong call unready.
- Queue fullness deliberately leaves readiness true.

**Verification:** T10 proves blocked worker inference does not block liveness or the
HTTP deadline. T11 proves shutdown refuses new work and bounds an active waiter.
T14 covers loading/draining plus watchdog and fatal-worker readiness. T15 runs five
success/cancellation/expiry waves without pending growth or an active batch leak.
The overload API test fills one running plus one pending slot, verifies the next
request gets 429 and `Retry-After: 1`, and confirms readiness remains 200.

## 2026-09-06 — M1 observability

**Purpose:** expose request, queue, batch, latency, and failure behavior without
unbounded labels or raw input.

- Added per-app Prometheus collectors for admission decisions, terminal outcomes,
  pending depth, active batches, actual batch size, server duration, queue wait,
  adapter execution duration, phase timing families, and execution failures.
- Added JSON logs with a fixed safe field set for lifecycle, errors, and shutdown.
- Added local `GET /metrics` and tests for its Prometheus output.

**Limitation:** the fake adapter has no meaningful preprocessing/forward/postprocess
phases. Preprocessing and postprocessing collectors have no observations until M2
defines real adapter phase timing. `inference_forward_duration_seconds` currently
measures the complete blocking fake adapter call and is named for the real-adapter
contract that follows.

## M1 completion status

**M1 gate: passed locally.** Verification environment: macOS on Apple Silicon,
uv 0.12.10, Python 3.12.14.

| Check | Result |
| --- | --- |
| `uv sync --locked` in a fresh temporary copy | Passed; 32 packages resolved |
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | Passed; 30 files |
| `uv run mypy` | Passed; 24 source files |
| `uv run pytest` | 48 passed in 0.92 seconds; 2 upstream warnings |
| Four simultaneous real HTTP predictions | 4 correct correlated responses |
| Prometheus batch observation | one dispatched batch, size sum 4 |
| Uvicorn Ctrl-C shutdown | draining and stopped events; clean process exit |

The HTTP smoke used the timed policy with a 100 ms collection window solely to
make the batch easy to observe. It is a functional demonstration, not a benchmark.
No throughput or latency claim is derived from it.

The two existing upstream TestClient/AnyIO deprecation warnings remain visible and
do not fail tests. Real adapters, training, benchmarks, Docker, CI, and portfolio
release work remain later milestones.

## 2026-09-07 — M2 model selection and preparation

**Purpose:** make external model acquisition explicit, pinned, and reproducible.

- Verified the official repository API and model card: DistilBERT sequence
  classification, English SST-2/GLUE, Apache-2.0, safetensors available.
- Resolved immutable Hub commit `714eb0fa89d2f80546fda750413ed43d93601a13`.
- Added an allowlisted preparation script for exactly five files and a generated
  SHA-256 manifest. Prepared size on this host: 256 MiB.
- Kept all artifacts under ignored `artifacts/`; no model weights entered Git.
- Added optional `hf` dependencies and refreshed the lockfile.

**Verification:** preparation completed without authentication and the manifest's
five files passed adapter integrity checks. Exact hashes are recorded in the model
card. The network warning only concerned anonymous Hub rate limits.

## 2026-09-07 — M2 real adapter and runtime integration

**Purpose:** run a genuine batch-capable model through the unchanged M1 scheduler.

- Added local-only tokenizer and sequence-classification loading with safetensors
  and remote custom code disabled.
- Validated artifact identity, hashes, tokenizer length, two labels, and label order.
- Added longest-item padding, 256-token truncation, attention masks, one tensorized
  forward pass, softmax interpretation, evaluation/inference mode, and CPU/CUDA
  device handling.
- Added adapter phase timings to the shared protocol and Prometheus metrics.
- Added `huggingface` startup configuration while preserving fake defaults.
- Added committed fake and Hugging Face environment profiles as inspectable examples;
  the application does not implicitly load them.

**Verification:** a two-item CPU smoke predicted positive for “I loved this movie.”
and negative for “This was terrible and boring.” This is a functional observation,
not a quality metric. CPU load, warmup, and close succeeded.

## 2026-09-07 — M2 correctness and offline restart

**Purpose:** prove real batching, parity, HTTP sharing, and reproducible local reload.

- Added six explicitly marked real-model tests.
- T12 compares three variable-length inputs alone and batched at absolute score
  tolerance `1e-6`.
- Forward instrumentation proves one three-item adapter batch uses one model call.
- The unchanged scheduler combines three real inputs and correlates expected labels.
- Three independent HTTP requests share one forward call and expose batch metrics.
- Two fresh adapters reload under Hub/Transformers offline flags and reproduce the
  same prediction within `1e-8`.

**Verification:** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --extra hf pytest
-m model` passed 6 tests with 48 deliberately deselected. The default offline gate
passed 48 tests with 6 model tests deliberately deselected. Both retain the two
documented upstream TestClient/AnyIO warnings.

**M2 gate: passed locally.** Independent HTTP requests demonstrably share one real
forward pass under the batch-capable configuration, and prepared artifacts reload
offline. CUDA was not available and is not claimed as tested. Training, the custom
adapter, quality evaluation, benchmarks, Docker, and CI remain later milestones.

Fresh-copy verification installed the full locked `hf` environment, passed Ruff,
strict mypy over 26 source files, the 48-test offline gate, and the 6-test explicit
model gate (10.94 seconds). A real Uvicorn process then accepted three simultaneous
localhost requests, returned positive/negative/positive in input order, and exposed
one observed batch with size sum three. Ctrl-C logged draining and stopped events
and completed application shutdown. This smoke is correctness evidence, not a
performance measurement.
