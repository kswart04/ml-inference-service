"""Audit retained raw outcomes and expose accepted-input mix alongside aggregate reports."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.load import TEXTS
from inference_service.adapters.custom_artifacts import write_json


def audit(path: Path) -> list[dict[str, object]]:
    report = json.loads(path.read_text())
    if "provenance" not in report:
        raise ValueError(f"Incomplete report: {path}")
    raw_dir = Path("benchmarks/raw") / Path(report["config"]["output"]).stem
    results: list[dict[str, object]] = []
    for run in report["results"]:
        name = f"{run['policy']}-{run['repetition']}-{run['rate_requested']}-{run['burst']}"
        outcomes: Counter[str] = Counter()
        mix: dict[str, Counter[int]] = defaultdict(Counter)
        count = sent = 0
        with gzip.open(raw_dir / f"{name}.jsonl.gz", "rt") as handle:
            for line in handle:
                row = json.loads(line)
                if row["index"] != count:
                    raise ValueError("Missing or duplicate raw arrival index")
                count += 1
                sent += row["sent_s"] is not None
                outcomes[row["outcome"]] += 1
                mix[row["outcome"]][row["index"] % len(TEXTS)] += 1
        if (
            count != run["intended"]
            or sent != run["sent"]
            or dict(outcomes) != run["outcomes"]
            or run["final_outstanding"] != 0
            or run["final_pending"] != 0
            or run["final_active_batches"] != 0
        ):
            raise ValueError(f"Aggregate/raw count or drain mismatch: {name}")
        results.append(
            {
                "run": name,
                "intended": count,
                "sent": sent,
                "outcomes": dict(outcomes),
                "input_index_by_outcome": {
                    outcome: dict(counts) for outcome, counts in mix.items()
                },
                "raw_count_and_drain_checks_passed": True,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/results/raw-audit.json"))
    args = parser.parse_args()
    reports = {str(path): audit(path) for path in args.inputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, reports)
    print(f"Audited {sum(map(len, reports.values()))} runs against retained raw request records")


if __name__ == "__main__":
    main()
