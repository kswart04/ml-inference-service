import gzip
import json
from pathlib import Path

import pytest

from benchmarks.audit import audit


def test_raw_audit_rejects_an_aggregate_that_hides_a_client_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw_dir = Path("benchmarks/raw/example")
    raw_dir.mkdir(parents=True)
    with gzip.open(raw_dir / "single-0-1.0-False.jsonl.gz", "wt") as handle:
        handle.write(json.dumps({"index": 0, "sent_s": 0.0, "outcome": "200"}) + "\n")
        handle.write(
            json.dumps({"index": 1, "sent_s": None, "outcome": "client_capacity_drop"}) + "\n"
        )
    run = {
        "policy": "single",
        "repetition": 0,
        "rate_requested": 1.0,
        "burst": False,
        "intended": 2,
        "sent": 1,
        "outcomes": {"200": 1, "client_capacity_drop": 1},
        "final_outstanding": 0,
        "final_pending": 0,
        "final_active_batches": 0,
    }
    report = {"provenance": {}, "config": {"output": "example.json"}, "results": [run]}
    path = Path("example.json")
    path.write_text(json.dumps(report))
    assert audit(path)[0]["raw_count_and_drain_checks_passed"]
    run["outcomes"] = {"200": 2}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="mismatch"):
        audit(path)
