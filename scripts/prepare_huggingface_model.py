from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from inference_service.adapters.huggingface import (
    MANIFEST_NAME,
    MODEL_INTERNAL_VERSION,
    MODEL_LICENSE,
    MODEL_MAX_LENGTH,
    MODEL_REPO_ID,
    MODEL_REVISION,
    REQUIRED_FILES,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        local_dir=output_dir,
        allow_patterns=sorted(REQUIRED_FILES),
    )
    missing = sorted(name for name in REQUIRED_FILES if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Prepared snapshot is missing required files: {missing}")
    manifest = {
        "schema_version": 1,
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "internal_version": MODEL_INTERNAL_VERSION,
        "license": MODEL_LICENSE,
        "max_length": MODEL_MAX_LENGTH,
        "files": {name: sha256(output_dir / name) for name in sorted(REQUIRED_FILES)},
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned Hugging Face SST-2 model.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/huggingface-sst2"))
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(f"Prepared {MODEL_REPO_ID}@{MODEL_REVISION} and wrote {manifest}")


if __name__ == "__main__":
    main()
