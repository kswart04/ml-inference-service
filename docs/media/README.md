## Architecture

- Image: [inference-architecture.png](inference-architecture.png)
- Vector: [SVG](inference-architecture.svg) · [PDF](inference-architecture.pdf)
- Title: **Inference Service Request Flow**
- Description: Shows how the API validates requests, admits them to a bounded queue,
  forms batches, runs model inference, and returns each result to the correct caller.
- Alt text: Architecture diagram of a FastAPI inference service. The event loop owns
  request validation, a bounded queue, batching, deadlines, and response routing. A
  dedicated worker runs one batch at a time using a custom PyTorch model or pinned
  DistilBERT. Ordered results return to the event loop and each original caller.

## Model evaluation

- Image: [custom-model-results.png](custom-model-results.png)
- Vector: [SVG](custom-model-results.svg) · [PDF](custom-model-results.pdf)
- Title: **Custom Sentiment Model: Held-Out Results**
- Description: A PyTorch classifier trained from scratch reached 68.82% accuracy on
  449 held-out UCI sentences, compared with a 50.11% majority-class baseline.
- Alt text: Model evaluation card showing 68.82% test accuracy, an improvement of
  18.71 percentage points over the majority-class baseline, and macro-F1 of 0.6811.
  The confusion matrix has 188 correct negative and 121 correct positive predictions,
  with 37 negatives predicted positive and 103 positives predicted negative.
  Evaluation uses 449 held-out UCI sentiment sentences.

The architecture follows [ARCHITECTURE.md](../ARCHITECTURE.md). The results come
from [custom-small.json](../results/custom-small.json); they describe the recorded
small-profile model and test split, not a new training run. The baseline always
predicts negative, using the tie rule fixed from the balanced training set.

Dataset: Kotzias (2015), *Sentiment Labelled Sentences*, UCI Machine Learning
Repository, [DOI 10.24432/C57604](https://doi.org/10.24432/C57604), CC BY 4.0.

## Regenerate

From the repository root with the benchmark dependencies installed:

```bash
MPLCONFIGDIR=/tmp/ml-inference-media-matplotlib \
uv run --extra benchmark python scripts/create_portfolio_media.py
```

The script reads the saved evaluation results and writes both graphics in all three
formats. It checks the confusion-matrix totals, accuracy, and text placement before
exporting. Layout and wording can be edited in the script.
