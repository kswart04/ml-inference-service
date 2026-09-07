from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

from inference_service.adapters.custom_artifacts import sha256, tokens, write_json

URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
ARCHIVE_SHA256 = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"
DOMAINS = ("amazon_cells", "imdb", "yelp")


class Row(TypedDict):
    id: str
    text: str
    label: int
    domain: str


def split_rows(rows: list[Row], seed: int) -> tuple[dict[str, list[Row]], int]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        groups[" ".join(tokens(row["text"]))].append(row)
    unique = [
        group[0]
        for key, group in sorted(groups.items())
        if key and len({r["label"] for r in group}) == 1
    ]
    strata: dict[tuple[str, int], list[Row]] = defaultdict(list)
    for row in unique:
        strata[row["domain"], row["label"]].append(row)
    rng = random.Random(seed)
    splits: dict[str, list[Row]] = {"train": [], "validation": [], "test": []}
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        train_end, validation_end = int(len(group) * 0.7), int(len(group) * 0.85)
        splits["train"].extend(group[:train_end])
        splits["validation"].extend(group[train_end:validation_end])
        splits["test"].extend(group[validation_end:])
    for split in splits.values():
        rng.shuffle(split)
    return splits, len(rows) - len(unique)


def distribution(rows: list[Row]) -> dict[str, object]:
    return {
        "count": len(rows),
        "labels": dict(Counter(str(r["label"]) for r in rows)),
        "domains": dict(Counter(r["domain"] for r in rows)),
    }


def read_split(directory: Path, name: str) -> list[Row]:
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{name}.jsonl"
    if sha256(path) != manifest["splits"][name]["sha256"]:
        raise ValueError(f"Split integrity check failed: {name}")
    result: list[Row] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result.append(Row(id=row["id"], text=row["text"], label=row["label"], domain=row["domain"]))
    return result


def prepare(archive: Path, output: Path, seed: int = 42) -> None:
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Dataset archive does not match pinned SHA-256.")
    rows: list[Row] = []
    with zipfile.ZipFile(archive) as bundle:
        for domain in DOMAINS:
            raw = bundle.read(f"sentiment labelled sentences/{domain}_labelled.txt").decode()
            domain_rows: list[Row] = []
            # IMDb includes Unicode line separators inside sentences, not record boundaries.
            for number, line in enumerate(raw.rstrip("\n").split("\n"), 1):
                line = line.rstrip("\r")
                text, label = line.rsplit("\t", 1)
                if label not in {"0", "1"} or not text.strip():
                    raise ValueError("Unexpected dataset row")
                domain_rows.append(
                    Row(id=f"{domain}:{number}", text=text.strip(), label=int(label), domain=domain)
                )
            if Counter(r["label"] for r in domain_rows) != {0: 500, 1: 500}:
                raise ValueError("Unexpected source distribution")
            rows.extend(domain_rows)
    splits, removed = split_rows(rows, seed)
    output.mkdir(parents=True, exist_ok=False)
    details = {}
    for name, records in splits.items():
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        details[name] = {**distribution(records), "sha256": sha256(path)}
    write_json(
        output / "manifest.json",
        {
            "source": URL,
            "source_sha256": ARCHIVE_SHA256,
            "license": "CC-BY-4.0",
            "citation": "Kotzias (2015), Sentiment Labelled Sentences, UCI, doi:10.24432/C57604",
            "seed": seed,
            "raw_count": len(rows),
            "removed_duplicate_or_conflicting_rows": removed,
            "split_method": "canonical-token deduplication; domain/label-stratified 70/15/15",
            "splits": details,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the pinned CC BY 4.0 UCI dataset.")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/uci-sentiment"))
    args = parser.parse_args()
    archive = args.archive
    if archive is None:
        archive = Path("data/uci-sentiment.zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
    prepare(archive, args.output)
    print((args.output / "manifest.json").read_text())


if __name__ == "__main__":
    main()
