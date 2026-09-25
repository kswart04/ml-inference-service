# CPU serving experiments

## Setup

These experiments compare the three implemented policies on the two prepared
real models. Model quality is evaluated separately in the model cards. All runs
used one developer workstation and synthetic inputs of mixed lengths. Response
caching was disabled.

The baseline matrix uses batch cap 8, timed window 5 ms, pending capacity 32,
server deadline 2 seconds, and client timeout 5 seconds. Single-item mode always
dispatches one item despite the cap. One server runs at a time, with two PyTorch
intra-op threads and one inter-op thread. Models use CPU float32 and inference mode.
The client is a separate Python process on the same host using loopback HTTP.

Hardware: Apple M5 Pro, 18 logical CPUs, 48 GiB RAM, macOS 26.6.2 arm64, Python
3.12.14, PyTorch 2.14.0. No GPU was used. Linux container checks use the separately
locked CPU-only PyTorch wheel. The macOS performance measurements do not establish
Linux container performance.

## Workload and client validity

`benchmarks.load.TEXTS` contains six authored English inputs, cycled in fixed order.
They range from a short sentiment word to repeated review sentences crossing the
256-token truncation limit. Each real model applies its own documented tokenizer,
maximum length, and longest-item padding. Exact strings and character lengths are
recorded in every completed report. This corpus measures serving behavior, not quality.

The client uses absolute intended arrival times. It does not wait for one
response before scheduling the next request. It permits at most 256 outstanding
tasks and 256 HTTP connections. An arrival more than 100 ms late is recorded as
a client late drop; a full client budget causes a client capacity drop. Dropped
arrivals are not rescheduled. Each raw record includes intended/send/completion
times, scheduling lag, outcome, and HTTP duration. Burst traffic compresses each
half-second's arrivals into the first 100 ms, preserving the exact intended
count at 5× on-rate.

A run is **client-invalid** if any client drop occurs or p99 scheduling lag exceeds
20 ms. Such runs remain visible in charts and tables but cannot establish server
capacity. Client CPU/RSS is recorded alongside server CPU/RSS. CPU percentages use
psutil's per-process convention: 100% means one fully occupied logical CPU, and a
multithreaded server can exceed 100%.

Warmup sends two groups of eight requests and is excluded. Loading and download
time are excluded. Each run drains its HTTP tasks and then waits for pending/active
server work to clear, so timed-out native work cannot contaminate the next run.
The experiment fails if server work does not drain.

## Pilot and frozen target

The short pilots are retained as `docs/results/pilot-*.json`. DistilBERT single-item
mode handled low loads, while high load produced 429 responses. Padding mixed input
lengths caused batching to reduce capacity in some pilot conditions. The custom
model's highest pilot saturated the client, which was recorded as invalid.

The pilots informed the target used for all subsequent comparisons:
**successful-response p95 ≤ 100 ms and unsuccessful intended arrivals ≤ 1%, with a
valid client**. This target applies only to these experiments.
`configs/benchmark.json` records the target, rate selections, and other frozen settings.

The steady matrix uses three 60-second repetitions for every model/policy/rate:

| Model | Below-capacity candidate | Near-capacity candidate | Overload attempt |
| --- | ---: | ---: | ---: |
| Custom | 200 req/s | 800 req/s | 1,600 req/s |
| DistilBERT | 20 req/s | 80 req/s | 160 req/s |

The rate categories come from the pilots; capacity varies by policy. Burst
checks use three 15-second repetitions, paired with steady traffic at the same
mean rate (custom 800, DistilBERT 80). Their shorter durations demonstrate
transient behavior and are not equivalent to a long sustained-capacity
experiment. Policies run sequentially in single/immediate/timed order;
repetition and rate order are recorded. Thermal drift and other activity on a
shared workstation can affect measurements. They are not randomized trials on
isolated hardware.

## Reproduction

Prepare both artifacts as described in the README, then install the benchmark extra:

```bash
uv sync --locked --extra hf --extra custom --extra benchmark
uv run --no-sync python -m benchmarks.experiment --adapter huggingface \
  --rates 20 80 160 --duration 60 --repetitions 3 --output artifacts/cpu-huggingface.json
uv run --no-sync python -m benchmarks.experiment --adapter custom \
  --rates 200 800 1600 --duration 60 --repetitions 3 --output artifacts/cpu-custom.json
uv run --no-sync python -m benchmarks.experiment --adapter huggingface \
  --rates 80 --duration 15 --repetitions 3 --bursts --output artifacts/burst-huggingface.json
uv run --no-sync python -m benchmarks.experiment --adapter custom \
  --rates 800 --duration 15 --repetitions 3 --bursts --output artifacts/burst-custom.json
```

Each output must be new. The runner starts/stops its own local server on port 8765;
stop any other server on that port first. Do not run the two models' experiments
simultaneously on this host. Allow about an hour for this matrix, plus preparation.
For a quick driver check use a fresh output path, `--duration 2 --repetitions 1`
and `--adapter fake --rates 50`. Use the longer runs for performance comparisons.

Regenerate tables and standalone SVG/PNG plots from completed reports:

```bash
MPLCONFIGDIR=/tmp/ml-inference-matplotlib \
uv run --no-sync python -m benchmarks.report \
  artifacts/cpu-custom.json artifacts/cpu-huggingface.json \
  --output artifacts/benchmark-report
```

Pass the burst files separately to another output directory to avoid pooling the
60-second steady matrix with the shorter burst checks' control runs. Table quantiles
are means of each repetition's quantiles, not pooled percentiles. Throughput is
successful completions within the intended load window per second. Reports also
retain successful throughput including HTTP drain time, per-outcome durations,
counts, latency quantiles, scheduling lag, final outstanding work, resource samples,
Prometheus deltas, and source/lock/artifact fingerprints.

## Reading the results

Prometheus batch-size buckets are cumulative: subtract adjacent buckets for ranges;
the mean is size sum divided by count. Queue wait and adapter phase means likewise
use their histogram sums/counts. Client end-to-end latency includes HTTP overhead;
it is not interchangeable with forward-pass time. Mean forward duration is per
batch, while queue wait is per request. A longer batch forward can serve several
requests, so compare throughput and constraints together.

Fast rejections can lower average latency while useful throughput falls. The report
shows latency for successful requests, counts for every outcome, and client-validity
flags. The target must hold in every repetition to be marked met.

## DistilBERT results

For the baseline snapshot, single-item mode met the target in all three repetitions
at both 20 and 80 requests/sec. At 80 requests/sec it completed 80.0 successful
requests/sec, with mean per-run p95 27.20 ms and no unsuccessful arrivals. Immediate
batching at that rate varied between 52.7 and 80.0 successful requests/sec and
rejected 16.63% of intended arrivals across the repetitions. Timed batching averaged
61.1 successful requests/sec with 22.80% unsuccessful arrivals.

At low load, the timed policy's mean p95 was 37.05 ms versus single-item's 30.79 ms.
Mean queue wait was about 6.1 ms with timed collection and 0.1 ms with single-item
dispatch. This illustrates the cost of deliberate waiting when no batch forms.

At 160 requests/sec, single-item mode completed about 93.0 successful requests/sec,
while immediate batching completed about 49.4. Neither met the latency/error target.
Mean dispatched batch size was about 7.9 under batching. Mixed-length padding is
consistent with the increased batch execution cost; these results do not establish
that batching loses for homogeneous lengths or on a GPU.

Two DistilBERT intervals had client late drops (one immediate/20 interval and one
timed/160 interval). They are retained and marked client-invalid. The single-item
and immediate/160 comparisons above use valid clients; no capacity claim is made
from those invalid intervals.

## Custom-model results

At 200 requests/sec, immediate and timed policies met the target in all three
runs. Single-item throughput/latency was similar, but one run had 26 late client
drops and is therefore not a clean all-repetition result. At 800 requests/sec,
both single and immediate completed all 144,000 intended arrivals across three
repetitions with mean per-run p95 of 1.15 ms and 1.12 ms respectively. Their
observed mean batch size was 1.0: arrivals were handled quickly enough that
immediate queue snapshots rarely contained neighbors. The server kept up at this
rate, but these runs did not locate its capacity limit.

The timed policy at 800 requests/sec formed mean batches of 3.61, yet all runs
were client-invalid and only 62.79% of intended arrivals succeeded. The server
used less CPU, while system/client scheduling fell behind. Because both
processes share the same workstation, this does not establish a timed-policy
server limit or prove the cause. Larger batches and lower server CPU did not
translate into more successful requests.

All custom runs at 1,600 requests/sec saturated the client budget, across every
policy. The reports retain those runs to show where the load generator fell
behind. They cannot establish whether the server could have accepted a different
client implementation at that rate.

## Burst findings

The 15-second DistilBERT controls at mean 80 requests/sec were client-valid. Single
and immediate met the target under steady traffic. During 5× bursts, no run was
client-invalid, but no policy met the 100 ms latency target. Single-item mode
completed about 79.9 requests/sec with 0.14% unsuccessful arrivals and mean p95
332.25 ms. Immediate and timed batching completed about 47.4 requests/sec, rejected
about 39%, and had mean p95 above 800 ms. The mixed-length full batches were again
more expensive than individual forwards on this CPU.

All custom burst experiments at mean 800 requests/sec were client-invalid, as were
most paired 15-second steady controls. They show that this same-host client cannot
generate that burst schedule reliably and support no server comparison. The valid
60-second steady custom results above remain the applicable evidence.

## Reports and limitations

The [steady table](results/benchmark/table.md) and
[steady chart](results/benchmark/cpu-comparison.svg) contain all 54 repeated steady
settings. The [burst table](results/burst/table.md) and
[burst chart](results/burst/cpu-comparison.svg) show paired controls and bursts.
Black crosses mark any setting containing a client-invalid repetition. Dashed lines
in the burst chart are burst traffic. Compact group summaries retain resources,
mean internal timings, mean batch size, all outcomes, and min/max repetition bounds.

Across 90 runs, `benchmarks.audit` reconciled every aggregate against every retained
raw arrival index and outcome. All completed runs ended with zero client tasks,
server pending requests, and active batches. The
[audit result](results/raw-audit.json) preserves input index by outcome, while raw
per-request timing files remain ignored because they are bulky.

The workload token lengths are 1, 9, 14, 36, 144, 256 for the custom tokenizer and
4, 12, 18, 42, 158, 256 including DistilBERT special tokens. Exact versions and
lengths are in [workload.json](results/workload.json). Server RSS ranged by model and
policy; detailed peaks and process CPU samples are in `summary.json`. Process CPU
is observational on a shared host and was not normalized into a cross-machine score.

On this CPU and mixed-length sequence, batching did not improve DistilBERT
capacity. Timed collection was also unnecessary for the very cheap custom model
at the valid 800 requests/sec steady rate. A follow-up experiment could group
requests by length, with starvation protection and the same client-validity
checks.

The experiment reports fingerprint all benchmark/service Python sources, lockfile,
model manifest, workload, and hardware. The runner originally sampled Git HEAD at
completion; M4 commits advanced while the long matrix ran, so that field reads
`2f9963b` even though the authoritative source hashes were captured at experiment
start. Later runner code records the start commit explicitly. DistilBERT's report
began before NumPy was added to the custom-only dependency extra and therefore has
the earlier lock hash; the other reports use the current lock hash. Neither change
altered inference code used by the measured server. The deadline/cancellation fix
was developed in an isolated worktree and integrated only after measurements, so
the report describes the pre-fix scheduler snapshot. That fix affects terminal error
translation and child-task cleanup, not successful inference or batching decisions.
