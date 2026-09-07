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
