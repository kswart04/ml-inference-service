# ML Inference Service

An educational text classification backend with a custom batching scheduler
planned across incremental milestones. See [the requirements](docs/REQUIREMENTS.md).

**Current milestone: M1.** The API runs a deterministic fake adapter through the
custom bounded scheduler. There is no trained model or measured performance result yet.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the
repository root:

```bash
uv sync --locked
uv run uvicorn inference_service.api.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

The project uses Python 3.12 (`uv` can install it automatically). No model downloads,
GPU libraries, credentials, or external services are needed for the fake adapter.

In another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8000/health/live
curl --fail-with-body http://127.0.0.1:8000/health/ready
curl --fail-with-body http://127.0.0.1:8000/v1/models
curl --fail-with-body http://127.0.0.1:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"fake-sentiment","model_version":"v1","input":{"text":"The acting was excellent."}}'
```

Interactive API documentation is at <http://127.0.0.1:8000/docs>. Stop with Ctrl-C.

The example returns `label: positive` with a fixed positive score of `0.8`, a
negative score of approximately `0.2`, and a new server request ID. These values
are fake fixtures. [Adapter behavior](docs/ARCHITECTURE.md#fake-adapter-semantics)
describes the exact rule and tie handling.

## Configuration and boundaries

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_ADAPTER` | `fake` | Only adapter currently accepted |
| `INFERENCE_MAX_BODY_BYTES` | `32768` | Per-request HTTP body cap before JSON parsing |
| `INFERENCE_MAX_TEXT_CHARACTERS` | `8000` | Text character cap before prediction |
| `INFERENCE_SCHEDULING_POLICY` | `timed` | `single`, `immediate`, or `timed` |
| `INFERENCE_MAX_BATCH_SIZE` | `8` | Maximum items dispatched together |
| `INFERENCE_MAX_COLLECTION_DELAY_MS` | `10` | Timed policy's oldest-item window |
| `INFERENCE_PENDING_CAPACITY` | `128` | Waiting items per configured model |
| `INFERENCE_REQUEST_DEADLINE_MS` | `5000` | Handler-entry to terminal deadline |
| `INFERENCE_GRACEFUL_SHUTDOWN_SECONDS` | `10` | Drain allowance |
| `INFERENCE_WORKER_WATCHDOG_SECONDS` | `30` | Running-batch stuck threshold |
| `INFERENCE_RETRY_AFTER_SECONDS` | `1` | `Retry-After` value for queue overload |

Numeric limits must be positive integers. Settings load at application creation;
invalid settings fail startup. The app does not automatically load a `.env` file.
Requests require both `model_id` and `model_version`; the configured identity is
`fake-sentiment/v1`.

The API implements safe errors for oversized bodies (413), invalid input (422),
unknown models/versions (404), queue overload (429), unavailable workers/draining
(503), deadlines (504), and execution errors (500). Responses carry `X-Request-ID`;
prediction/error bodies include the same ID. No traceback or submitted text is
returned in error details.

The event loop owns admission, queues, futures, deadlines, and terminal states. A
single dedicated thread owns blocking adapter work, with one submitted/running
batch. `/metrics` exposes Prometheus text locally; application logs are structured
JSON and exclude input text. Use one local server process because multiple workers
would create independent queues and model copies.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

These checks use the fake adapter and make no model or dataset downloads. The
test suite covers M0 contracts and input boundaries, not the complete M1
concurrency acceptance suite. Runtime package installation requires network
access initially; tests run offline after dependencies are installed.

## Following milestones

| Milestone | Work remaining |
| --- | --- |
| M2 | Pinned Hugging Face artifact preparation, adapter, real parity tests |
| M3 | CPU training, held-out evaluation, custom artifacts, reload tests |
| M4 | Benchmarks and plots, Docker, CI, clean-checkout release verification, demo |

Real-model setup, training, and benchmark commands will be documented when they
exist. GPU installation is deferred until a real-model milestone; CPU is the
required baseline.

## Project documentation

- [Work log](docs/WORK_LOG.md): every completed section and its verification.
- [Architecture](docs/ARCHITECTURE.md): contracts and request flow.
- [Decisions](docs/DECISIONS.md): choices and their trade-offs.
- [Learning notes](docs/LEARNING_NOTES.md): concepts to understand before M1.

Repository licensing has not yet been selected by the owner.
