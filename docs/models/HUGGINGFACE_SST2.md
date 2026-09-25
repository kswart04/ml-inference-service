# Hugging Face SST-2 adapter card

## Identity and provenance

| Field | Value |
| --- | --- |
| Hub repository | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| Pinned commit | `714eb0fa89d2f80546fda750413ed43d93601a13` |
| Internal API version | `hf-sst2-714eb0fa-max256` |
| Architecture | DistilBERT sequence classification |
| Training task/data | English binary sentiment, SST-2/GLUE |
| Upstream license | Apache-2.0 |
| Framework | PyTorch through Transformers |

The source [model
card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
reports upstream evaluation results and limitations. Those quality measurements
have not been repeated here. The model card notes that examples can produce
biased predictions; review it before downstream use.

## Preparation and integrity

`scripts/prepare_huggingface_model.py` is the only supported downloader. Its repo,
commit, and allowed files are constants rather than caller arguments. It downloads
the README, configuration, safetensors weights, tokenizer configuration, and
vocabulary. It writes `inference-service-manifest.json` with the exact identity,
preprocessing version, license identifier, and SHA-256 for every required file.

The adapter refuses missing, changed, or wrongly identified files. It calls
`from_pretrained()` with a local path, `local_files_only=True`,
`trust_remote_code=False`, and `use_safetensors=True`. Prediction requests cannot
provide a repository, revision, path, URL, or remote-code option.

The artifact directory is ignored by Git. Its generated manifest recorded these hashes:

| File | SHA-256 |
| --- | --- |
| `README.md` | `75f17282923fea5b541df92640e2d8646c4ff5e8e3e16ed0a7e4e50ee7588851` |
| `config.json` | `582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15` |
| `model.safetensors` | `7c3919835e442510166d267fe7cbe847e0c51cd26d9ba07b89a57b952b49b8aa` |
| `tokenizer_config.json` | `5ab9097b4149371c5fd52b2d6e26cb6f9c07c0d19fcfdda895b1adad6b57c3e0` |
| `vocab.txt` | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |

## Input and output interpretation

Input is one nonblank text string per HTTP request, capped at 8,000 characters by
default. The tokenizer receives a whole scheduler batch at once with:

- `padding="longest"`
- `truncation=True`
- `max_length=256`, including special tokens
- attention masks and PyTorch tensors

The model runs one forward call for the batch under `torch.inference_mode()` in
evaluation mode. Softmax converts its two logits into scores. Upstream label ID 0
must be `NEGATIVE` and ID 1 must be `POSITIVE`; startup fails otherwise. Scores sum
approximately to one but are not presented as calibrated confidence.

CPU is the verified baseline. CUDA can be selected only when PyTorch reports it
available. CUDA forward timing synchronizes before the timer stops. CUDA out of
memory becomes a fatal worker signal and requires restart.

## Verified behavior and limitations

M2 tests compare single-item predictions with predictions beside substantially
different input lengths. Labels match and both scores use absolute tolerance
`1e-6`. Forward-call counters check that a three-item adapter batch and three independent
HTTP requests each use one model forward call. Two newly constructed adapters load
with Hugging Face offline flags and reproduce the same prediction within `1e-8`.

This model targets English sentiment from SST-2. The custom model was trained on
UCI sentences from Amazon, IMDb, and Yelp, so comparing their raw accuracy would
mix dataset and architecture differences. This card covers loading and prediction
checks; serving benchmarks are in [BENCHMARKS.md](../BENCHMARKS.md).
