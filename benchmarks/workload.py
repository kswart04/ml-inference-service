"""Describe the exact prepared-token lengths used by the benchmark corpus."""

import argparse
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

from benchmarks.load import TEXTS
from inference_service.adapters.custom_artifacts import CustomConfig, encode, write_json


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom", type=Path, default=Path("artifacts/custom-sentiment"))
    parser.add_argument("--hf", type=Path, default=Path("artifacts/huggingface-sst2"))
    parser.add_argument("--output", type=Path, default=Path("docs/results/workload.json"))
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.hf, local_files_only=True, trust_remote_code=False
    )
    hf_tokens = tokenizer(list(TEXTS), padding=False, truncation=True, max_length=256)["input_ids"]
    config = CustomConfig.model_validate_json((args.custom / "config.json").read_text())
    words = json.loads((args.custom / "vocabulary.json").read_text())
    indices = {word: i for i, word in enumerate(words)}
    cpu_name = platform.processor()
    if platform.system() == "Darwin":
        cpu_name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    report = {
        "cpu_name": cpu_name,
        "versions": {
            name: version(name)
            for name in ("torch", "transformers", "httpx", "psutil", "matplotlib")
        },
        "inputs": [
            {
                "index": i,
                "characters": len(text),
                "utf8_bytes": len(text.encode()),
                "custom_tokens_after_truncation": len(encode(text, indices, config.max_length)),
                "hf_tokens_including_specials_after_truncation": len(hf_tokens[i]),
            }
            for i, text in enumerate(TEXTS)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
