# ML Inference Service with Dynamic Batching

Original design, 6 September 2026. Development history is in [WORK_LOG.md](WORK_LOG.md).

## 1. Purpose

Build a small, well-tested inference backend that combines compatible online prediction requests into batches. Use a custom scheduler with a PyTorch model trained from scratch and a pretrained Hugging Face model. Measure how batching affects throughput, latency, and overload.

The project focuses on concurrency, bounded resource use, request lifecycles, model integration, and reproducible experiments.

Another developer should be able to install the service, run the demo, prepare both models, run tests, and reproduce the benchmarks from the repository instructions.

## 2. Scope and priorities

MUST marks a requirement, SHOULD a desirable addition, and MAY an optional extension. Each milestone implements part of the scope below.

### Required release scope

- Python HTTP API for stateless text classification.
- Custom batching scheduler with maximum batch size and maximum collection delay.
- Three scheduling policies: single-item, immediate available-item batching, and timed dynamic batching.
- Explicit model adapter interface and versioned startup registry.
- Two real adapters: a custom trained PyTorch classifier and a Hugging Face sequence-classification model.
- Bounded pending queues, bounded running work, request deadlines, result correlation, cancellation cleanup, and controlled shutdown.
- Health/readiness endpoints, structured logs, metrics, and meaningful tests.
- CPU operation as the required development baseline; CUDA benchmarking if hardware is available.
- Reproducible model training and serving benchmarks, with separate reports for model quality and system performance.
- Docker, CI, setup documentation, architecture decisions, and a demonstration script.

### Out of scope

The initial service runs in memory. Kubernetes, Kafka, Redis, PostgreSQL, accounts, billing, a frontend, autoscaling, cloud provisioning, hot model swapping, streaming generation, arbitrary model uploads, and distributed worker recovery are out of scope.

LLM token generation and continuous batching are out of scope. They involve requests completing at different generation steps and often managing KV caches; they need a different scheduling design. Image, audio, embeddings, and tabular adapters are possible future extensions.

## 3. What model-agnostic means here

The scheduler MUST operate on request envelopes and adapter capabilities without knowing a particular neural network architecture. Model-specific validation, tokenization, tensor preparation, prediction, and output interpretation belong in adapters.

This does not mean that any downloaded model works automatically. A supported model needs a compatible adapter, output schema, batching policy, device support, and sufficient memory. Different modalities need additional input schemas. Some models cannot batch usefully.

Initially, support the same task across two architectures: binary text sentiment classification. This keeps the interface small and tests scheduler reuse across two architectures.

The two adapters MUST be tested through the same scheduler code. They may run in separate server sessions on limited hardware. Concurrent multi-model operation is optional; if enabled, models have isolated queues and version identities. Never mix their inputs into one batch.

## 4. Architecture and execution ownership

Use Python, FastAPI, Pydantic, PyTorch, Transformers, pytest, an asynchronous HTTP client for load generation, and Prometheus-compatible metrics. Resolve compatible supported dependency versions at implementation time and commit a lockfile. Prefer uv if available. Document CPU and optional CUDA installation separately.

Use one ASGI server process in the initial implementation. The asynchronous event loop owns queues, scheduler state, admission decisions, request futures, deadlines, and terminal-state transitions. Multiple Uvicorn workers would each have separate queues and model copies.

Run blocking tokenization and model execution outside the event loop using a dedicated bounded executor. Start with a single execution thread and at most one submitted/running batch per configured adapter. Do not allow an executor's internal queue to become a second unbounded request queue. The model is loaded once, put in evaluation mode, and used under inference mode inside the execution thread.

On a shared GPU, default to one active model per server session. Concurrent model execution requires an explicit device-level capacity policy; separate per-model semaphores alone do not bound total GPU work.

The worker returns ordered results or a batch error. The event loop alone resolves client futures. Each submitted item has a unique server-generated request ID and a strong association with its future. Never rely on response completion order to identify the client.

Trade-off: a thread cannot safely terminate a hung native model call. Request deadlines can free clients, but execution remains occupied. A stuck worker marks readiness false after a configured watchdog threshold; recovery requires a process restart. Hard worker termination and automatic recovery require a process-based worker and supervisor, which are deferred.

## 5. Adapter contract

Define typed structures rather than passing arbitrary dictionaries throughout the core. An interface can expose these responsibilities; exact method names may change if documented:

| Responsibility | Required behavior |
| --- | --- |
| Metadata | Stable model ID, immutable version, task, supported input type, labels, device, and maximum batch size |
| Load | Load trusted local artifacts once; validate tokenizer, label mapping, configuration, and weights |
| Validate | Check cheap input structure and size before admission |
| Batch compatibility | Identify model/version and relevant options that can share one execution |
| Predict batch | Prepare inputs, execute the model once for the batch, and return one result per input in the same order |
| Close | Release resources during orderly shutdown |

The batch method MUST perform one batched model forward pass, rather than one forward pass per input. Preprocessing loops may be necessary and are permitted.

Text preprocessing MUST specify padding, truncation, maximum sequence length, attention masks where required, and label interpretation. Reject empty/whitespace-only inputs. Apply a byte-level HTTP body limit before parsing and a character limit before queue admission. Adapter maximum token length provides a separate bound on tensor size.

Use padding to the longest sequence in a batch within a fixed maximum length. The custom classifier must mask padding when pooling. A prediction on one input should agree with its batched prediction within documented numeric tolerance. Test this with varying neighboring input lengths.

Load external models from an operator-controlled allowlist with a pinned revision. Prediction callers cannot supply Hub IDs, file paths, arbitrary code, or download URLs. Model downloads occur in an explicit preparation step rather than a prediction request. Prefer safetensors and keep remote custom code disabled for the selected model.

## 6. Model plan

### Model A: custom classifier

Implement a compact PyTorch sentiment classifier: token embedding, masked mean pooling, a small hidden layer, and a two-class output. Initialize and train all weights locally, without pretrained embeddings.

Use a documented public sentiment dataset such as IMDb, subject to verifying its dataset card, availability, and permitted use before download. Record its exact source/revision, split construction, sample counts, preprocessing, and seed. Fit vocabulary on training data only. Select hyperparameters using validation data. Keep test data untouched until final evaluation. Never use test samples to fit vocabulary or make tuning decisions.

Provide a bounded training configuration suitable for initial CPU experimentation and a fuller configuration. Record training wall time and hardware. Export weights, vocabulary, tokenizer settings, architecture configuration, label map, seed, artifact hash, and evaluation metrics. Keep large artifacts outside ordinary Git history with a reproducible retrieval or training procedure.

Report accuracy, macro F1, class distribution, and a majority-class baseline. The trained model must beat the majority baseline on held-out data. If it falls short, investigate and record the result.

### Model B: Hugging Face pretrained classifier

Initial candidate: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`. Verify the model card, license, required libraries, label map, and exact revision during implementation. Load using the appropriate tokenizer and sequence-classification model class. Batch tensors directly so scheduler ownership remains visible.

Use an explicit internal version tied to the resolved Hub commit and preprocessing configuration. Record source attribution. This model was trained for a different dataset/domain from an IMDb-trained custom classifier; do not treat raw accuracy differences as a controlled comparison of architecture quality.

Benchmark scheduling policies within each model independently. Throughput differences between the two models do not measure batching improvement. The small custom model may show little or no benefit; report that result too.

### Development order

Start with a deterministic fake adapter to exercise concurrency without downloads. Add a real pretrained model next, then the custom training pipeline. Both real adapters are required for the final release, but model training must not block initial backend development.

## 7. API contract

### POST /v1/predict

One input per HTTP request in version one. Server-side batches combine independent requests. Clients do not specify batch membership or scheduling policy.

Example request:

```json
{"model_id":"custom-sentiment","model_version":"v1","input":{"text":"The acting was excellent."}}
```

Example response shape (values are illustrative, not model output):

```json
{"request_id":"server-generated-id","model_id":"custom-sentiment","model_version":"v1","prediction":{"label":"positive","scores":{"negative":0.12,"positive":0.88}}}
```

Scores are model scores/probabilities as defined by the adapter, not calibrated confidence claims. The response MUST report the exact version used. Require explicit versions initially; defer mutable aliases such as latest.

### Other endpoints

- `GET /health/live`: process/event-loop liveness; never run inference here.
- `GET /health/ready`: required model loaded, scheduler active, worker not failed/stuck, service not draining. Temporary queue fullness alone is not a readiness failure.
- `GET /v1/models`: configured models, versions, task, and availability; omit local paths or credentials.
- `GET /metrics`: Prometheus-compatible metrics; local/development exposure initially.

### Errors

| Status | Meaning |
| --- | --- |
| 413 | HTTP body too large |
| 422 | Invalid schema, empty text, or configured character limit exceeded |
| 404 | Unknown model ID or version |
| 429 | Pending queue capacity reached; include a documented Retry-After value |
| 503 | Model unavailable, worker failed, or service draining |
| 504 | Accepted request exceeded server deadline |
| 500 | Unexpected execution or internal error |

Use a consistent error envelope with request ID, stable error code, and safe message. Do not return tracebacks. Request IDs are for tracing, not idempotency: retries are new prediction attempts. There is no durable delivery or exactly-once guarantee.

## 8. Scheduling and request lifecycle

### Proposed initial defaults

These are development settings, not performance recommendations. Make them configurable and validate ranges at startup.

| Setting | Initial value |
| --- | --- |
| Maximum batch size | 8 |
| Maximum collection delay | 10 ms |
| Pending capacity per active model | 128 requests |
| Maximum in-flight batches per model | 1 |
| Server request deadline | 5,000 ms |
| Maximum request body | 32 KiB |
| Maximum text characters | 8,000 |
| Maximum text tokens | 256, bounded by model support |
| Graceful shutdown allowance | 10 seconds |
| Worker watchdog | 30 seconds |

Start the deadline at handler entry, before admission. The body cap limits parsing work; the reverse proxy/network path is outside this server deadline. Use monotonic clocks for durations and wall time only for human-readable logs.

### Policies

1. **Single-item:** dispatch exactly one live item whenever the worker is free.
2. **Immediate batching:** when free, dispatch up to the maximum size from live items already waiting, without deliberate additional wait.
3. **Timed batching:** when free, dispatch a full batch immediately; otherwise wait only until the oldest pending item's collection window expires, or new arrivals fill the batch.

If the oldest item already waited beyond the collection window while the worker was occupied, dispatch a partial batch as soon as the worker becomes free. Do not start a fresh full waiting window. Maximum collection delay is not a total queue-delay or response-time guarantee.

Apply FIFO among compatible live requests. Remove expired/cancelled entries before forming batches. Keep dispatch timing independent of deadlines; expire requests according to the contract and report those outcomes. Deadline-aware scheduling is a later policy extension.

Keep all pending items, including items considered for a future batch, within the pending-capacity accounting. At most one active batch adds up to max_batch_size items beyond that bound. Preprocessing starts only for the active batch. Dead queue entries must be reclaimed promptly so cancellations cannot cause persistent false overload or growing tombstones.

### Terminal states and cancellation

Represent accepted requests as pending, running, succeeded, failed, expired, or cancelled. A request resolves once; late completion never changes a terminal state. Admission rejection is counted separately from accepted work.

- Pending timeout/disconnect: remove the item and release its queue resources.
- Running timeout/disconnect: resolve or cancel the waiter and discard its eventual output. Other batch members must still receive correct outputs.
- Native execution continues after a caller times out; do not free the execution slot early.
- Batch exception: fail remaining live members and preserve scheduler health if the error is recoverable. Do not retry automatically in version one.
- CUDA out-of-memory or unusable worker: fail the batch, mark unavailable, and require documented restart/recovery. Do not retry automatically.
- Result-count mismatch: fail the batch as an adapter contract violation rather than misrouting outputs.

Graceful shutdown stops admission, marks not ready, and allows accepted work to finish within the grace period and its existing deadlines. Afterwards fail remaining live waiters and exit under the process supervisor's bounded termination policy. A hung thread cannot be forcibly killed safely; document this limitation.

## 9. Observability

Use bounded-cardinality labels: configured model/version, scheduling policy, and a small outcome enumeration. Never use request IDs or input text as metric labels. Do not log raw user text by default.

Required metrics: accepted and rejected request counts; terminal outcomes; pending depth; active batches; actual dispatched batch size; server request duration; queue wait; preprocessing duration; forward-pass duration; postprocessing duration; execution failures; expired/cancelled work.

Define queue wait as admission to worker start. Keep preprocessing separate from model forward time. Client-measured end-to-end latency includes network overhead and is the primary benchmark latency. CUDA forward measurements must synchronize appropriately or use CUDA events; otherwise asynchronous execution can produce misleading times. Report synchronization methodology and use the same instrumentation for all compared policies.

Log request IDs, model versions, error categories, and shutdown events. Optional Grafana dashboards should visualize queue growth, throughput, latency percentiles, and errors together.

## 10. Required correctness tests

Use a controllable fake adapter and an injectable monotonic clock or synchronization primitives for scheduler tests. Avoid relying on exact millisecond sleeps or tight timing thresholds in shared CI. Add separate real-time smoke tests with generous bounds.

| ID | Acceptance test |
| --- | --- |
| T01 | Concurrent distinct inputs get the correct corresponding results even when clients time out |
| T02 | A full batch dispatches without waiting for the entire collection window |
| T03 | A partial batch dispatches after its oldest-item window expires |
| T04 | Busy-worker wait is not followed by a second unnecessary collection window |
| T05 | Queue admission is atomic under concurrency and never exceeds capacity |
| T06 | Pending expiration/disconnect reclaims capacity and does not execute abandoned work |
| T07 | Running expiration does not free the execution slot or break surviving batch members |
| T08 | Different model versions cannot mix within a batch |
| T09 | Batch exception and wrong output count resolve all affected live waiters exactly once |
| T10 | Slow inference does not block health endpoints or event-loop deadlines |
| T11 | Shutdown refuses new work and gives existing waiters bounded outcomes |
| T12 | Both real adapters agree between single and batched predictions within declared tolerance |
| T13 | Oversized/invalid input fails before queue admission; invalid config fails startup |
| T14 | Readiness reflects loading, draining, and worker failure correctly |
| T15 | Repeated success, cancellation, and timeout waves leave no orphan futures or queue growth |
| T16 | A fresh process reloads custom exported artifacts and reproduces test predictions |

Always run offline fake-adapter tests in CI. Make downloaded-model tests explicit, cached where appropriate, and separately marked. Tests with missing artifacts must report a skip or setup failure. No GPU is required for the default CI gate.

## 11. Benchmark design

The scheduler must pass the correctness tests before benchmarking. A throughput improvement is not required.

Record Git commit, lockfile, OS, CPU, RAM, optional GPU and driver/runtime, device, thread counts, precision, model artifact revision/hash, tokenizer configuration, input-length distribution, scheduling settings, and client placement.

Compare all three policies for each model on the same hardware and workload. Disable response caching. Warm up before measurement and exclude model loading/download time; report cold start separately if measured. Keep the model in evaluation/inference mode for every policy.

Start with a small pilot to determine sustainable arrival rates. Then evaluate below-capacity traffic, near-capacity traffic, overload, and bursts. Sweep batch sizes such as 1, 4, 8, 16 and windows such as 0, 2, 5, 10 ms selectively; do not run a huge matrix before identifying useful ranges. Zero window is valid and should behave like immediate batching.

Use an open-loop arrival schedule for capacity and latency experiments: the intended arrival rate should not silently slow down when the server becomes slow. Bound client outstanding work, record intended arrivals, actual sends, scheduling lag, client-side drops, response outcomes, and final outstanding counts. A saturated client invalidates a server-capacity conclusion. Closed-loop concurrency tests may supplement the report but must be labeled separately.

For each selected configuration, target three measured repetitions after warmup, initially about 60 seconds each if feasible. Extend only where sample counts or variability require it. Record exact durations. Report successful requests/sec, attempted load, p50/p95/p99 client latency for successful requests, timeout/rejection rates, queue wait, batch-size distribution, and resource use. Report outcome counts and durations alongside latency percentiles for successful requests.

Choose an explicit latency target after the baseline, freeze it for comparative experiments, and label it as a project target. A throughput gain is meaningful only alongside the same latency/error constraints. Report variability, input characteristics, and counterexamples where batching hurts. Separate model quality evaluation from serving performance. Retain raw compact result files and plotting scripts.

## 12. Repository layout and developer experience

Use a src-based Python package. Suggested paths:

- `src/inference_service/api/`: routes, validation, error translation.
- `src/inference_service/core/`: envelopes, scheduler, admission, lifecycle.
- `src/inference_service/adapters/`: protocol, fake, custom, Hugging Face.
- `src/inference_service/runtime/`: configuration, registry, executor, startup/shutdown.
- `src/inference_service/observability/`: logs and metrics.
- `training/`: dataset preparation, training, evaluation, export.
- `benchmarks/`: load driver, scenarios, result schema, plotting.
- `tests/`: unit, integration, and optional model tests.
- `configs/`: fake, custom, Hugging Face, and benchmark profiles.
- `docs/`: this specification, architecture, decisions, model cards, benchmark report.
- `scripts/`: explicit artifact preparation and demonstration entry points.
- Root: README, pyproject, lockfile, Dockerfile, gitignore, CI workflow.

README commands must be tested from a clean environment. Include quick start with fake adapter, real-model preparation, API request example, custom training, test suite, and a short benchmark. Provide CPU Docker operation first. Bind locally by default. Setup must run locally without paid resources.

Ignore environments, credentials, caches, raw datasets, large weights, and bulky benchmark traces. Keep small aggregate measurements and configuration needed to reproduce them. Keep third-party attribution and record code, model, and dataset licenses separately. The code license is undecided.

## 13. Implementation milestones and completion gates

### M0 — Repository and contracts

Set up package configuration, README, test harness, typed request/result structures, adapter protocol, fake adapter, and a basic prediction/health API. Establish baseline lint/type/test commands. No real model download or training yet.

Gate: service runs locally; a fake prediction and basic invalid-input tests pass; batching is deferred to M1. Document planned architecture and any dependency decisions.

### M1 — Scheduler and bounded lifecycle

Implement the three policies, single-worker executor, queue limits, deadlines, cancellation, error propagation, metrics basics, and shutdown. Complete applicable fake-adapter tests T01–T11 and T13–T15.

Gate: correctness demonstrated under concurrency, timeouts, overload, and recoverable batch errors; no unbounded executor backlog.

### M2 — Hugging Face adapter

Add pinned model preparation, tokenizer/label mapping, inference execution, configuration, warmup, and real-model smoke/parity tests.

Gate: real independent HTTP requests share one forward pass under a batch-capable configuration and receive correct results; offline restart works with prepared artifacts.

### M3 — Custom model

Implement reproducible training, held-out evaluation, artifact export, model card, and custom adapter. Add reload and parity tests.

Gate: trained model beats documented trivial baseline; artifacts reload independently; scheduler core is unchanged to integrate the adapter.

### M4 — Experiments and portfolio release

Run CPU baseline experiments and optional GPU runs. Produce reproducible results, charts, architecture decisions, Docker setup, CI, and a short demo including overload behavior.

Gate: setup works from a clean checkout, and benchmark reports include hardware, errors, and enough detail to reproduce the measurements.

### Optional later work

Length-aware batching with starvation protection; embeddings or image adapter; process-isolated worker with supervised recovery; authenticated hosted demo; multi-model resource scheduling; comparison with Ray Serve or Triton; adaptive waiting policies. Add one only after M4 is complete.

## 14. Completion checklist

- Both real models run through the same scheduling core.
- Real batched forward passes, response routing, lifecycle tests, and overload behavior are demonstrated.
- Custom training and held-out evaluation are reproducible.
- Model versions, dependencies, configurations, and benchmark hardware are recorded.
- Results report successful throughput together with latency and failures.
- README works from a clean checkout; CI passes its documented offline gate.
- A demo shows baseline versus batching and a controlled overload scenario.
- Architecture decisions explain alternatives and limitations.
- Supported adapters, retry behavior, worker recovery limits, and measured performance are documented.

## 15. References

Background on tokenization and batching. Defaults and acceptance criteria above are specific to this project.

- Hugging Face padding/truncation: https://huggingface.co/docs/transformers/main/pad_truncation
- Initial pretrained model card: https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english
- NVIDIA Triton dynamic batcher, as a reference implementation: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
- Ray Serve batching, as a reference implementation: https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html

This document records the original requirements. See [README.md](../README.md) for current setup and [WORK_LOG.md](WORK_LOG.md) for completed work.
