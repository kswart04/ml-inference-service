# Project overview

A local inference service with a custom scheduler, two sentiment models, and CPU
benchmarks. The repository is private, with no hosted deployment or code license.

## Implemented

- Implemented a FastAPI inference service with a custom bounded batching scheduler,
  single-item/immediate/timed policies, deadlines, queue rejection, cancellation,
  shutdown behavior, and Prometheus-compatible metrics.
- Integrated two real binary-sentiment architectures through the same adapter
  contract: pinned DistilBERT and a PyTorch embedding/masked-mean classifier trained
  locally from initialization. Model integration did not require scheduler changes.
- Achieved 68.82% held-out accuracy and 0.6811 macro-F1 on the documented UCI sentence
  split, above a 50.11% training-majority accuracy baseline. The model card
  includes the confusion matrix and validation/test gap.
- Verified fresh-process artifact reload, mixed-length single/batch parity, and
  independent HTTP requests sharing one forward pass.
- Built an open-loop CPU experiment driver that retains scheduling lag, client
  drops, server errors, latency, throughput, batching metrics, and resource use.
- Added CI and non-root CPU container operation, including a read-only custom-model
  mount and real HTTP prediction checks.

Results, source revisions, and measurement files are in
[BENCHMARKS.md](BENCHMARKS.md). Runs where the load generator fell behind are marked
invalid for server-capacity comparisons.

## Limitations

Only the documented adapters are supported. Retries create new requests, and a hung
native call requires a process restart. GPU performance and production availability
have not been measured. Distributed serving, authentication, autoscaling, and public
deployment are outside the current scope. Batching was slower in several of the
mixed-length CPU experiments.

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
   Length grouping would also need a rule to prevent long requests from starving.
