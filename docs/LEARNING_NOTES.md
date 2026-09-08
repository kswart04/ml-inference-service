# Learning notes

## M0 — Contracts before scheduling

Read `core/contracts.py`, then `adapters/protocol.py`, then `api/app.py`.
The API knows which model identity was requested. The adapter knows how to
interpret text and produce results. The future scheduler will know when compatible
requests should execute; it will not know the neural network architecture.

Three design decisions to understand before M1:

1. **Execution ownership:** `async def` does not make synchronous code nonblocking.
   Real tokenization and model calls would occupy the event-loop thread, delaying
   health checks and deadlines. M1 moves those calls to a dedicated bounded worker.
   The current fake is intentionally small and temporary.
2. **Request lifetime versus execution lifetime:** timing out a client cannot
   safely interrupt a native thread already running inference. Keep the execution
   slot occupied until the worker finishes; otherwise the next batch could overlap
   and defeat the resource bound.
3. **Correlation and immutable versions:** associate each accepted request's ID
   with its own future. Preserve ordered batch inputs and outputs, verify output
   count, and resolve futures only on the event loop. Never route results by the
   order in which HTTP clients finish, or combine different model versions.

## Questions the experiments must answer later

Batching can improve utilization while increasing time spent waiting. Report
throughput together with latency, timeouts, and rejections. Compare policies within
the same model and workload. Fixed fake scores cannot demonstrate model quality,
and fake timings cannot establish real-model batching gains.

For timed batching, imagine a request has already waited 20 ms behind a busy
worker and the collection window is 10 ms. When the worker becomes free, the
scheduler should dispatch available work immediately; it should not wait another
10 ms. This becomes acceptance test T04.

## M1 — Follow one request through the scheduler

Admission happens while holding the scheduler condition lock, so checking capacity
and appending are atomic relative to other submissions. The request stays counted
as pending while the timed policy considers it. The driver removes only live FIFO
items, marks them running, and submits exactly one batch to the worker.

The HTTP coroutine waits on that envelope's future under its monotonic deadline.
If it expires or disconnects while pending, cleanup removes it. If it is running,
the future terminates but the worker continues. When the worker returns, the event
loop checks output count before pairing result index 0 with input envelope 0, and so
on. A terminal envelope is never resolved again.

Queue capacity bounds waiting work. One active batch can add at most the configured
batch size beyond that count. A full queue produces 429 and `Retry-After`; it does
not make readiness false. Draining, a stopped driver, watchdog expiry, or a fatal
worker signal makes readiness false.

## M2 — Model preparation is separate from serving

The Hub repository and revision live in adapter code, not the request schema. The
preparation script resolves that immutable snapshot into local files and hashes
them. Startup verifies identity and content before Transformers reads anything.
Serving uses offline local loading, so a prediction cannot cause a download or
execute repository-provided Python.

The scheduler still sees only `TextInput`, compatibility identity, and ordered
`Prediction` values. The adapter turns all texts into one rectangular tensor batch,
runs one forward call, then converts each logits row back into the same position.
This is the concrete proof that model integration did not change scheduling code.

Padding affects numeric operations, so parity means scores agree within a declared
tolerance rather than requiring identical bits. Testing short text beside long text
is important: it exercises padding and attention masks rather than comparing two
batches with the same shape.

## M3 — Quality, reproducibility, and model boundaries

Vocabulary construction is learning: even without labels, fitting it on test text
would expose held-out information. The small profile fits only its selected 1,200
training rows, uses validation for checkpoint selection, and opens test data only
after exporting a frozen checkpoint. The training-majority rule is fixed before
test scoring; it does not inspect test labels to choose its prediction.

An embedding padding row alone does not fix mean pooling. If a two-token sentence
is padded to ten tokens and divided by ten, its representation shrinks. Explicitly
mask both the summed embeddings and the token count. The padding test deliberately
sets the padding embedding to a large value to detect an incomplete implementation.

Reproducibility has distinct checks: two training processes produced identical
weights/vocabulary/configuration; a fresh serving process loaded those artifacts
offline and reproduced predictions; variable-length batching preserved scores to
`1e-6`. The manifest includes run provenance, so identical weights do not imply an
identical version when the recorded run timing differs.

Passing the majority baseline is a minimum quality gate, not a claim of a strong
sentiment system. The test confusion matrix exposes 103 missed positive examples,
and test accuracy falls below validation. Documenting that gap is more useful than
repeatedly tuning against the same test set. Serving latency and throughput still
need separate M4 experiments.

## M4 — Treat the experiment client as part of the measurement

A rate of 800 requests/sec means an intended arrival every 1.25 ms. Waiting for a
response before scheduling the next request would change the experiment when the
server slows. Absolute schedules preserve intended load, but the client must still
report how late it sent and whether it ran out of its own outstanding budget.

The timed window bounds deliberate collection, not response latency. Queue wait,
tokenization, model computation, and network overhead all contribute. A full batch
can have worse CPU cost when one long sentence forces many short sentences to pad
to 256 tokens. This explains why batching should be measured instead of assumed
to improve performance.

Rejections and client drops mean different things. HTTP 429 demonstrates the
server's admission rule; a client capacity drop means the generator never sent
that arrival. Neither should vanish from the denominator. A low latency percentile
for successful requests is only useful alongside those failures and the offered rate.

Clean optional environments matter too: a dependency installed for Transformers
can mask a missing dependency in custom-only training. A container also runs under
a different UID, so a valid local artifact can still fail on file permissions.
CI now exercises these boundaries directly rather than treating a Dockerfile as
proof that the container works.
