# Portfolio evidence and limits

The repository remains private. “Release” here means a documented, tested local
project; it does not imply a public hosted service, registry publication, or a
repository license selected on the owner's behalf.

## Claims supported by implementation

- Implemented a FastAPI inference service with a custom bounded batching scheduler,
  single-item/immediate/timed policies, deadlines, queue rejection, cancellation,
  shutdown behavior, and Prometheus-compatible metrics.
- Integrated two real binary-sentiment architectures through the same adapter
  contract: pinned DistilBERT and a PyTorch embedding/masked-mean classifier trained
  locally from initialization. Model integration did not require scheduler changes.
- Achieved 68.82% held-out accuracy and 0.6811 macro-F1 on the documented UCI sentence
  split, above a 50.11% training-majority accuracy baseline. This is a modest
  educational classifier, not a state-of-the-art sentiment result.
- Verified fresh-process artifact reload, mixed-length single/batch parity, and
  independent HTTP requests sharing one forward pass.
- Built an open-loop CPU experiment driver that retains scheduling lag, client
  drops, server errors, latency, throughput, batching metrics, and resource use.
- Added CI and non-root CPU container operation, including a read-only custom-model
  mount and real HTTP prediction checks.

Final benchmark findings belong in [BENCHMARKS.md](BENCHMARKS.md), with exact source
revisions and measurement files. Do not convert a client-invalid load attempt into
a server-capacity claim or describe a pilot as the repeated comparison.

## Claims this project does not support

No universal model compatibility, exactly-once delivery, automatic native-hang
recovery, measured GPU speedup, production availability/SLA, distributed serving,
authentication, autoscaling, or public deployment is implemented or demonstrated.
There is no requirement to claim a throughput improvement: batching can lose on
this mixed-length CPU workload.

## Interview walkthrough

1. Follow one request from body validation through queue admission to its future.
   Explain which state belongs to the event loop and what runs on the worker thread.
2. Show how a timeout ends the caller's wait while native inference keeps the
   execution slot occupied. Explain why a process restart is needed for a hung call.
3. Show T12/T16 and the independent-HTTP forward-count test. Explain why padding
   requires masking and why parity uses a numeric tolerance.
4. Trace training-only vocabulary fitting, validation checkpoint selection, and the
   frozen held-out evaluation. Discuss false negatives and the validation/test gap.
5. Run the short overload demo. Distinguish HTTP 429 from client drops and inspect
   unsuccessful outcomes alongside successful latency.
6. Explain a measured batching counterexample before proposing length-aware grouping.
   Such an extension also needs fairness/starvation rules, not just faster averages.
