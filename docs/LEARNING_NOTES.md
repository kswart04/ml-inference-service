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
