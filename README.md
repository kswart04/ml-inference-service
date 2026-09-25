# ML Inference Service

A FastAPI text classification service with a custom batching scheduler. It supports
single-request, immediate-batch, and timed-batch execution through the same API.

Adapters are available for a deterministic test fixture, pinned Hugging Face
DistilBERT, and a small PyTorch classifier trained from scratch. The custom model
reached 68.82% held-out accuracy against a 50.11% majority baseline. The repository
includes CPU benchmarks, Docker setup, CI, and an overload demo. See the
[requirements](docs/REQUIREMENTS.md) for the original design and milestones.

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
are fake fixtures. [Adapter
behavior](docs/ARCHITECTURE.md#model-identity-and-fake-adapter) describes the
exact rule and tie handling.

## Hugging Face model setup

Install the optional ML dependencies and download the pinned model snapshot:

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
| `INFERENCE_ADAPTER` | `fake` | `fake`, prepared `huggingface`, or trained `custom` |
| `INFERENCE_ARTIFACT_DIR` | `artifacts/huggingface-sst2` | Operator-controlled local model path |
| `INFERENCE_CUSTOM_ARTIFACT_DIR` | `artifacts/custom-sentiment` | Operator-controlled custom export |
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

Capacity and size limits must be positive; collection delay may be zero.
Settings load at application creation;
invalid settings fail startup. The app does not automatically load a `.env` file.
Requests require both `model_id` and `model_version`; discover the configured
identity at `/v1/models`. The default is `fake-sentiment/v1`.

The API returns errors for oversized bodies (413), invalid input (422),
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
uv run pytest
```

These checks require no model or dataset downloads. The default suite covers API
contracts, scheduler concurrency and lifecycle, and data/tokenizer/metric logic.
Package installation requires network access initially; tests run offline afterward.
Strict type checking of all code requires the optional ML libraries' type information:

```bash
uv run --extra hf --extra custom --extra benchmark mypy
```

The default test run excludes tests that require model artifacts. After preparing
the artifacts, test both real models offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --extra hf --extra custom pytest -m model
```

Tests skip with a reason when artifacts are missing. To test only one prepared
adapter, pass `tests/test_huggingface_adapter.py` or
`tests/test_custom_adapter.py` as well.

## Train and serve the custom model

From the repository root, prepare the hash-pinned UCI Sentiment Labelled Sentences
dataset (CC BY 4.0), train with the small CPU profile, and evaluate the frozen export:

```bash
uv sync --locked --extra custom
uv run --extra custom python -m training.data
uv run --extra custom python -m training.train --profile small
uv run --extra custom python -m training.evaluate --output artifacts/custom-evaluation.json
INFERENCE_ADAPTER=custom \
uv run --extra custom uvicorn inference_service.api.app:create_app \
  --factory --host 127.0.0.1 --port 8000 --workers 1
```

Only data preparation downloads anything. All subsequent model operations use
local files. Preparation and training refuse existing output directories;
evaluation refuses to replace an existing evaluation. For another experiment use
`--output artifacts/custom-another-run` when training, then point evaluation's
`--artifact` and serving's `INFERENCE_CUSTOM_ARTIFACT_DIR` at that directory.
For evaluation of another export, also pass a fresh `--output` report path. The
checked-in `docs/results/custom-small.json` preserves the original measurement.

The `full` profile uses all 2,076 training examples and permits 50 epochs; `small`
uses 1,200 and permits 25. Both select the best validation macro-F1 checkpoint with
seven-epoch patience. The full profile has not been quality-evaluated. Do not tune
new experiments against the already-reported test scores.

Call `/v1/models` to obtain `custom-sentiment` and the generated `custom-…` version,
then supply that exact pair to the same `/v1/predict` endpoint shown above. A rerun
has a different provenance report and thus a different version, even when its
weights reproduce exactly. Configuration examples are in `configs/custom.env`.
See the [custom model card](docs/models/CUSTOM_SENTIMENT.md) for the split recipe,
architecture, exact hashes, measured hardware, quality limits, and reproduction.

## Quick benchmark check

To check the benchmark driver with the fake adapter:

```bash
uv run --extra benchmark python -m benchmarks.experiment \
  --adapter fake --rates 50 --duration 2 --repetitions 1 \
  --output artifacts/benchmark-smoke.json
```

The [benchmark report](docs/BENCHMARKS.md) covers the full experiment settings and
results. The [operating guide](docs/OPERATIONS.md) covers CPU Docker commands, CI,
and the real-model overload demo.

CPU has been tested. CUDA support is implemented but has not been tested on this host.

## Project documentation

- [Work log](docs/WORK_LOG.md): development history and check results.
- [Architecture](docs/ARCHITECTURE.md): contracts and request flow.
- [Decisions](docs/DECISIONS.md): choices and their trade-offs.
- [Learning notes](docs/LEARNING_NOTES.md): concepts behind each completed milestone.
- [Hugging Face adapter card](docs/models/HUGGINGFACE_SST2.md): provenance and behavior.
- [Custom model card](docs/models/CUSTOM_SENTIMENT.md): training and held-out results.
- [CPU experiments](docs/BENCHMARKS.md): workload, validity rules, and reproduction.
- [Operation and demo](docs/OPERATIONS.md): containers, CI, and overload behavior.

Repository licensing has not yet been selected.
