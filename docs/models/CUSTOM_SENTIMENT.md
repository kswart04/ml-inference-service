# Custom sentiment classifier

This is a small English binary-sentiment classifier trained locally from random
initialization, with no pretrained embeddings or weights. It uses the same
scheduler as DistilBERT and serves as a small CPU baseline. Held-out accuracy is
68.82%; the errors are broken down below.

## Dataset, attribution, and split

Source: Dimitrios Kotzias (2015), *Sentiment Labelled Sentences*, UCI Machine
Learning Repository, DOI [10.24432/C57604](https://doi.org/10.24432/C57604). The
[official dataset
card](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
identifies the dataset as CC BY 4.0. Attribution is retained here and in the
local dataset manifest. Associated paper: Kotzias, Denil, de Freitas, and Smyth,
*From Group to Individual Labels Using Deep Features*, KDD 2015.

The official archive contains 1,000 sentences each from Amazon product reviews,
IMDb movie reviews, and Yelp restaurant reviews; each source has 500 examples per
class. This is UCI's sentence benchmark, not the Stanford 50,000-review IMDb dataset.
UCI was selected for its stated license and small size, which keeps CPU training short.

Archive URL:
`https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip`

Pinned SHA-256:
`afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`

`training.data` checks that hash and reads only three allowlisted ZIP members,
without extracting archive paths. It parses literal newline record boundaries and
the final tab label; Unicode line separators inside two IMDb sentences stay inside
the sentence. All three original 500/500 distributions are checked before splitting.

Before partitioning, normalize each text to its tokenizer's word sequence. For
identical sequences with consistent labels retain the first source row; remove all
rows in conflicting-label groups and empty-token groups. This removes 29 rows.
Sort canonical keys, group by domain and label, shuffle with Python Random(42),
and split each stratum at floor(0.70*n) and floor(0.85*n). Shuffle each final split
with that same RNG. IDs preserve source filename and original line number.

| Split | Negative | Positive | Total |
| --- | ---: | ---: | ---: |
| Available training | 1,038 | 1,038 | 2,076 |
| Selected small-profile training | 600 | 600 | 1,200 |
| Validation | 223 | 223 | 446 |
| Held-out test | 225 | 224 | 449 |

The small profile deterministically samples 200 rows per domain/label stratum
from training with seed 42. Vocabulary fitting uses **only these selected
training rows**. Preparation reads the original unsplit corpus to create
partitions; training reads only train and validation files, with their manifest
hashes verified. No test text is used to construct vocabulary, select an epoch,
or tune a hyperparameter. Final evaluation opens the test file after the export
is frozen, verifies provenance and ID separation, and writes an evaluation
record without changing model identity. Split file hashes and source
distributions are in the [aggregate result](../results/custom-small.json).

## Architecture and preprocessing

- Lowercase, then extract ASCII words with optional internal apostrophes using
  `[a-z]+(?:'[a-z]+)?`. This tokenizer drops numbers,
  punctuation, and non-ASCII characters; it does not interpret HTML or emoji.
- Retain the first 256 tokens. Vocabulary is frequency-descending with alphabetic
  tie-breaking, capped at 10,000 including PAD=0 and UNK=1. Actual size: 2,923.
- Unknown words map to UNK. Nonblank text that yields no tokens becomes one UNK.
  Empty/whitespace-only text is rejected by the API. Character cap is separately 8,000.
- Pad to the longest sequence in each batch. Embed each token into 64 dimensions,
  explicitly mask PAD out of the sum and denominator, then mean-pool.
- Apply a 64-unit linear layer, ReLU, and a two-unit linear output. Softmax yields
  negative (index 0) and positive (index 1) scores; ties choose positive in serving.
- One forward call handles the whole batch. Evaluation/inference mode is used;
  the same network and encoding functions are shared by training and serving.

The adapter supports batch sizes 1–32 and CPU/CUDA configuration. CUDA has **not**
been tested on this host. CUDA timing synchronizes before and after the forward
pass; CUDA OOM marks the worker unavailable. CPU is the measured execution device.

## Training and checkpoint selection

Both profiles use seed 42, float32 CPU, two PyTorch intra-op threads, deterministic
algorithms, AdamW (learning rate 0.003, weight decay 0.01), cross-entropy, and batch
size 32. Each epoch shuffles with a seeded torch Generator. There is no dropout.
The maximum validation macro-F1 selects the checkpoint; earliest epoch wins ties.
Stop after seven consecutive epochs without improvement.

| Profile | Training rows | Epoch ceiling | Observed execution |
| --- | ---: | ---: | --- |
| `small` | 1,200 | 25 | 19 epochs executed; epoch 12 selected |
| `full` | 2,076 | 50 | Available but not run or quality-evaluated |

One small configuration was selected before test evaluation. A second identical
small run checked determinism without accessing test examples. No test-driven
tuning was performed after seeing the results below.

Hardware: Apple M5 Pro, 18 logical CPUs, 48 GiB RAM, macOS 26.6.2 arm64. Python
3.12.14, PyTorch 2.14.0. No CUDA available; Apple MPS was not used. The first
training function recorded **0.892 seconds**, including split reading, fitting,
validation, and weight/config export, but excluding interpreter/library imports,
data download, and final provenance/manifest serialization. The second identical
run recorded 0.718 seconds under the same method. These timings cover training
on this dataset; serving performance is measured separately. Full package
versions are pinned in `uv.lock`.

## Held-out results

| Measurement | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Selected checkpoint, validation | 76.46% | 0.7631 |
| Selected checkpoint, test | **68.82%** | **0.6811** |
| Training-majority predictor, test | 50.11% | 0.3338 |

Training is balanced; the baseline tie rule chooses negative. Its prediction is
fixed from training labels before test evaluation. Macro-F1 includes both classes
and uses zero for an undefined class F1.

Test confusion matrix (rows true, columns predicted; negative then positive):

| | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| True negative | 188 | 37 |
| True positive | 103 | 121 |

The model beats the majority baseline, but misses 103 positive examples. Test
accuracy is 7.64 percentage points below validation. A small dataset and validation
checkpoint selection limit confidence in generalization. Scores are not calibrated
probabilities. Mean pooling discards word order, so negation and sarcasm are weak
points. There is no neutral class or multilingual support. Deduplication prevents
exact normalized-text overlap; source review/user IDs are unavailable, so related
sentences and near duplicates cannot be excluded by parent-review grouping.
This evaluation is not a controlled quality comparison against DistilBERT, which
was trained on a different dataset.

## Artifacts and reproduction

Default export: ignored `artifacts/custom-sentiment/`.

| File | Contents |
| --- | --- |
| `weights.safetensors` | Selected network state; no pickle loading |
| `vocabulary.json` | Ordered vocabulary and reserved IDs |
| `config.json` | Architecture, tokenizer version, dimensions, length, labels, seed |
| `training.json` | Training IDs, split hashes, history, metrics, hardware, code/lock hashes |
| `manifest.json` | Schema, exact allowlist, SHA-256 per file, derived version |
| `evaluation.json` | Frozen-artifact test report; added after selection |

Original serving identity: `custom-sentiment/custom-64a8f1b271531821`.
Version hashes all four manifest-listed files, including provenance. Timings make
the provenance of a retraining different, so the new version can differ even when
prediction-affecting artifacts match. Always discover your version via `/v1/models`.

Two independent training processes produced identical SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| Weights | `4370952bb5be63f6178df8d80334de28fda254d97e555aa341d94864e096d12d` |
| Vocabulary | `6579fb74de1ccf5ffefdd2d757a70cecfc8f56159ae76e1c2f03243b75179635` |
| Configuration | `5d304e14a64de6ac2900dec7eb2b188e929f6af82b86b88c22fcef20eb652136` |

These runs reproduced the same weights on this machine with the same lockfile.
Other PyTorch releases, platforms, or devices may produce different weights. See PyTorch's
[reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Follow the [README commands](../../README.md#train-and-serve-the-custom-model).
Preparation supports `--archive /path/to/pinned.zip` for offline reproduction.
Training accepts `--profile small|full`, `--dataset`, and a fresh `--output` path.
Evaluation accepts `--dataset`, `--artifact`, and a fresh `--output` report path.
Existing directories/evaluations are refused to preserve experiment history.
The original aggregate records the M2 base commit plus exact dirty-worktree source
hashes used for training; these identify the M3 implementation before its commit.

Serving validates the file allowlist, version, hashes, typed configuration,
vocabulary, and exact tensor state. It never downloads a model. T12 compares mixed
length single/batch predictions at absolute score tolerance `1e-6`; T16 starts a
new Python process in a different working directory, blocks socket connections,
loads the export, and reproduces predictions at `1e-6`. A separate HTTP test checks that
three independent requests share one custom forward pass with correct correlation.
The scheduler core has no M3 changes.

The code and trained weights do not yet have separate distribution licenses.
