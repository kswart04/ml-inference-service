# ML Inference Service

An educational text classification backend with a custom batching scheduler
planned across incremental milestones. See [the requirements](docs/REQUIREMENTS.md).

**Current milestone: M2.** The custom scheduler supports the deterministic fake and
a pinned Hugging Face DistilBERT SST-2 adapter. There is no personally trained model
or measured performance result yet.

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

## Hugging Face model setup

Install the optional ML dependencies and prepare the operator-pinned snapshot:

```bash
uv sync --locked --extra hf
uv run --extra hf python scripts/prepare_huggingface_model.py
```

The preparation command downloads five allowlisted files from
`distilbert/distilbert-base-uncased-finetuned-sst-2-english` at commit
`714eb0fa89d2f80546fda750413ed43d93601a13`, writes file hashes to a local manifest,
and stores about 256 MiB under the ignored `artifacts/` directory. Serving then
loads only that local directory with remote code and network fetching disabled.

Run the real adapter on CPU:

```bash
INFERENCE_ADAPTER=huggingface \
uv run --extra hf uvicorn inference_service.api.app:create_app \
  --factory --host 127.0.0.1 --port 8000 --workers 1
```

Its API identity is `huggingface-sentiment/hf-sst2-714eb0fa-max256`. Callers must
send that exact pair. Preparation is the only networked model step.
The equivalent committed startup values are in `configs/huggingface.env`; settings
are environment variables and are not loaded from that file automatically.

## Configuration and boundaries

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_ADAPTER` | `fake` | `fake` or prepared `huggingface` |
| `INFERENCE_ARTIFACT_DIR` | `artifacts/huggingface-sst2` | Operator-controlled local model path |
| `INFERENCE_DEVICE` | `cpu` | `cpu` or `cuda`; CUDA must be available |
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

The default gate excludes downloaded-model tests. After preparation, run the
explicit offline M2 gate:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --extra hf pytest -m model
```

## Following milestones

| Milestone | Work remaining |
| --- | --- |
| M3 | CPU training, held-out evaluation, custom artifacts, reload tests |
| M4 | Benchmarks and plots, Docker, CI, clean-checkout release verification, demo |

Training and benchmark commands will be documented when they exist. CPU is the
verified baseline; CUDA behavior is implemented but has not been tested on this host.

## Project documentation

- [Work log](docs/WORK_LOG.md): every completed section and its verification.
- [Architecture](docs/ARCHITECTURE.md): contracts and request flow.
- [Decisions](docs/DECISIONS.md): choices and their trade-offs.
- [Learning notes](docs/LEARNING_NOTES.md): concepts behind each completed milestone.
- [Hugging Face adapter card](docs/models/HUGGINGFACE_SST2.md): provenance and behavior.

Repository licensing has not yet been selected by the owner.
