# Local operation, containers, and demo

## CPU container

The image runs one Uvicorn process as UID 10001. It installs from `uv.lock`, uses
Python 3.12 and uv 0.12.10, and defaults to the fake adapter. It contains no model
weights, dataset, training tools, credentials, or development environment.
The build context is allowlisted in `.dockerignore`.

```bash
docker build -t ml-inference-service .
docker run --rm --name inference-service \
  -p 127.0.0.1:8000:8000 ml-inference-service
curl --fail http://127.0.0.1:8000/health/ready
```

The process binds to all interfaces *inside* the container so Docker port forwarding
works; the host port above is bound only to loopback. Stop with Ctrl-C or
`docker stop --time 20 inference-service`. The image health check uses readiness.
The Python base tag receives patch updates; this is a locked Python dependency
build, not a claim of bit-for-bit immutable OS image reconstruction.

For a prepared custom model, build its dependency extra and mount the local export
read-only. Run from the repository root after training:

```bash
docker build --build-arg MODEL_EXTRA=custom -t ml-inference-service:custom .
chmod a+r artifacts/custom-sentiment/weights.safetensors
docker run --rm --name inference-custom \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,source=$(pwd)/artifacts/custom-sentiment,target=/models/custom,readonly" \
  -e INFERENCE_ADAPTER=custom \
  -e INFERENCE_CUSTOM_ARTIFACT_DIR=/models/custom \
  ml-inference-service:custom
```

For DistilBERT use `MODEL_EXTRA=hf`, `INFERENCE_ADAPTER=huggingface`, mount
`artifacts/huggingface-sst2` at `/models/hf`, and set
`INFERENCE_ARTIFACT_DIR=/models/hf`. Files must be readable by container UID 10001.
Safetensors exports weights with owner-only permissions; the command above grants
read access to that public-dataset model file for the container's different UID.
The image loads them locally during startup. It never fetches weights on a request.
Linux's lock uses PyTorch's CPU wheel index, avoiding CUDA library downloads.

The fake container build and live HTTP prediction are verified in CI. Custom/HF
container configurations are provided for local use; their container execution is
not yet claimed as verified. CUDA is an untested extension: the committed Linux
environment is deliberately CPU-only. A GPU deployment requires an explicitly
resolved compatible CUDA wheel/driver environment and a new lock/benchmark record.

## CI boundaries

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual dispatch.
Actions are pinned by commit and receive read-only repository permission.

- `offline`: locked lightweight dependencies, Ruff lint/format, default fake/data/
  load-driver tests. No model or dataset is downloaded by this job.
- `types-and-custom`: optional ML/benchmark dependencies, strict mypy, pinned dataset
  preparation, bounded training, and five explicit custom artifact tests on macOS.
- `container`: Linux Docker build, readiness, and an actual HTTP prediction.
- `huggingface`: manual opt-in only; prepares the pinned snapshot and runs six
  model tests with offline flags. Run with
  `gh workflow run ci.yml -f huggingface=true`.

Dependency installation and explicit preparation require networking; offline tests
run after installation. CI custom training does not inspect the held-out test set
or overwrite the checked-in model quality result. It verifies the export it creates,
without assuming that a different CPU reproduces the original model version.

## Short batching and overload demo

Prepare the Hugging Face artifact using the README first. The following command
starts and stops its own local server for each policy, warms the model, compares
low load and overload, and records HTTP failures alongside successful latency:

```bash
uv sync --locked --extra hf --extra custom --extra benchmark
uv run --no-sync python -m benchmarks.experiment \
  --adapter huggingface --rates 20 160 --duration 5 --repetitions 1 \
  --pending-capacity 4 --output artifacts/demo.json
```

Use a fresh output name for another demo. Read each printed row's `policy`,
`outcomes`, `p95`, and `client_valid`; full metrics are in `artifacts/demo.json`.
HTTP 429 is a bounded-queue rejection, not a crash. Readiness should remain healthy;
the runner waits for pending and active work to clear after each load interval.
Raw request timings and server logs stay under `benchmarks/raw/demo/`.

Do not promise batching will improve this workload. Long and short inputs mixed
in a batch incur padding cost, and deliberate waiting can increase low-load latency.
The demo is a correctness demonstration; the repeated M4 report is performance evidence.

## Recovery and resource boundaries

Use one process per model session. The pending queue and active batch are bounded;
request deadlines stop waiting clients but cannot interrupt a native PyTorch call.
A hung worker makes readiness false and requires process restart. A process
supervisor or Docker's stop timeout can terminate the entire process after its
grace allowance. This service does not supervise or automatically restart hung
worker threads.

Health and metrics are local development endpoints. There is no authentication,
hosted public demo, or automatic scaling. The repository remains private.
