import json
from pathlib import Path

from benchmarks.report import summarize


def test_report_keeps_invalid_runs_and_error_denominator(tmp_path: Path) -> None:
    row = {
        "rate_requested": 10,
        "burst": False,
        "policy": "single",
        "success_latency_seconds": {"p50": 0.01, "p95": 0.02, "p99": 0.03},
        "successful_rps_in_window": 9,
        "intended": 10,
        "outcomes": {"200": 9, "429": 1},
        "metrics_delta": {"inference_batch_size_count": 3, "inference_batch_size_sum": 9},
        "client_valid": False,
        "scheduled_seconds": 1,
        "resources": [
            {
                "rss_bytes": 100.0,
                "cpu_percent": 1.0,
                "client_rss_bytes": 50.0,
                "client_cpu_percent": 2.0,
            }
        ]
        * 2,
        "final_pending": 0,
        "final_outstanding": 0,
    }
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps({"config": {"adapter": "custom"}, "provenance": {}, "results": [row]})
    )
    summary = summarize([path])[0]
    assert summary["unsuccessful_percent"] == 10
    assert summary["mean_batch_size"] == 3
    assert summary["invalid_client_runs"] == 1
    assert not summary["target_met_all_runs"]
    assert summary["outcomes"] == {"200": 9, "429": 1}
