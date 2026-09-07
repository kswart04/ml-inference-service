"""Regenerate aggregate tables and standalone charts from completed experiment JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, bool, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        report = json.loads(path.read_text())
        if "provenance" not in report:
            raise ValueError(f"Experiment is incomplete: {path}")
        for row in report["results"]:
            groups[
                (report["config"]["adapter"], row["rate_requested"], row["burst"], row["policy"])
            ].append(row)
    result = []
    for (adapter, rate, burst, policy), runs in sorted(groups.items()):
        p95 = [
            r["success_latency_seconds"]["p95"] * 1000
            for r in runs
            if r["success_latency_seconds"]["p95"] is not None
        ]
        p50 = [
            r["success_latency_seconds"]["p50"] * 1000
            for r in runs
            if r["success_latency_seconds"]["p50"] is not None
        ]
        p99 = [
            r["success_latency_seconds"]["p99"] * 1000
            for r in runs
            if r["success_latency_seconds"]["p99"] is not None
        ]
        rps = [r["successful_rps_in_window"] for r in runs]
        attempts = sum(r["intended"] for r in runs)
        successes = sum(r["outcomes"].get("200", 0) for r in runs)
        metrics: dict[str, float] = defaultdict(float)
        outcomes: dict[str, int] = defaultdict(int)
        for run in runs:
            for name, value in run["metrics_delta"].items():
                metrics[name] += value
            for name, count in run["outcomes"].items():
                outcomes[name] += count

        def mean_metric(prefix: str, values: dict[str, float] = metrics) -> float | None:
            count = values[prefix + "_count"]
            return values[prefix + "_sum"] / count if count else None

        invalid = sum(not r["client_valid"] for r in runs)
        result.append(
            {
                "adapter": adapter,
                "rate": rate,
                "burst": burst,
                "policy": policy,
                "repetitions": len(runs),
                "scheduled_seconds": sum(r["scheduled_seconds"] for r in runs),
                "intended": attempts,
                "outcomes": dict(outcomes),
                "invalid_client_runs": invalid,
                "successful_rps_mean": statistics.mean(rps),
                "successful_rps_min": min(rps),
                "successful_rps_max": max(rps),
                "successful_rps_stdev": statistics.stdev(rps) if len(rps) > 1 else 0.0,
                "p50_ms_mean": statistics.mean(p50) if p50 else None,
                "p95_ms_mean": statistics.mean(p95) if p95 else None,
                "p95_ms_min": min(p95) if p95 else None,
                "p95_ms_max": max(p95) if p95 else None,
                "p99_ms_mean": statistics.mean(p99) if p99 else None,
                "unsuccessful_percent": 100 * (attempts - successes) / attempts,
                "target_met_all_runs": invalid == 0
                and len(p95) == len(runs)
                and max(p95) <= 100
                and all(
                    (r["intended"] - r["outcomes"].get("200", 0)) / r["intended"] <= 0.01
                    for r in runs
                ),
                "mean_batch_size": mean_metric("inference_batch_size"),
                "mean_queue_wait_seconds": mean_metric("inference_queue_wait_seconds"),
                "mean_forward_seconds": mean_metric("inference_forward_duration_seconds"),
                "mean_preprocessing_seconds": mean_metric(
                    "inference_preprocessing_duration_seconds"
                ),
                "mean_postprocessing_seconds": mean_metric(
                    "inference_postprocessing_duration_seconds"
                ),
                "peak_server_rss_bytes": max(s["rss_bytes"] for r in runs for s in r["resources"]),
                "peak_client_rss_bytes": max(
                    s["client_rss_bytes"] for r in runs for s in r["resources"]
                ),
                "mean_server_cpu_percent": statistics.mean(
                    s["cpu_percent"] for r in runs for s in r["resources"][1:]
                ),
                "mean_client_cpu_percent": statistics.mean(
                    s["client_cpu_percent"] for r in runs for s in r["resources"][1:]
                ),
                "final_pending_max": max(r["final_pending"] for r in runs),
                "final_outstanding_max": max(r["final_outstanding"] for r in runs),
            }
        )
    return result


def charts(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"single": "#26547c", "immediate": "#008577", "timed": "#b45200"}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for i, adapter in enumerate(("custom", "huggingface")):
        for policy, color in colors.items():
            selected = sorted(
                [
                    r
                    for r in rows
                    if r["adapter"] == adapter and r["policy"] == policy and not r["burst"]
                ],
                key=lambda r: r["rate"],
            )
            if not selected:
                continue
            rates = [r["rate"] for r in selected]
            for j, (key, label) in enumerate(
                (
                    ("successful_rps_mean", "Successful requests/sec"),
                    ("p95_ms_mean", "Successful p95 latency (ms)"),
                    ("unsuccessful_percent", "Unsuccessful intended arrivals (%)"),
                )
            ):
                values = [r[key] for r in selected]
                axes[i, j].plot(rates, values, "o-", color=color, label=policy)
                for rate, value, row in zip(rates, values, selected, strict=True):
                    if row["invalid_client_runs"]:
                        axes[i, j].scatter(rate, value, marker="x", s=150, color="black", zorder=4)
                axes[i, j].set(
                    xlabel="Intended requests/sec",
                    ylabel=label,
                    title=f"{adapter} · steady traffic",
                )
                axes[i, j].grid(alpha=0.2)
            means = [r["successful_rps_mean"] for r in selected]
            axes[i, 0].errorbar(
                rates,
                means,
                yerr=[
                    [r["successful_rps_mean"] - r["successful_rps_min"] for r in selected],
                    [r["successful_rps_max"] - r["successful_rps_mean"] for r in selected],
                ],
                fmt="none",
                color=color,
                capsize=4,
            )
        axes[i, 1].axhline(100, color="gray", linestyle="--", linewidth=1)
        axes[i, 1].set_yscale("log")
        axes[i, 2].axhline(1, color="gray", linestyle="--", linewidth=1)
        axes[i, 0].legend()
    fig.suptitle(
        "CPU policy comparison · means of repetitions; throughput bars show min–max\n"
        "Black × marks client-invalid settings; these cannot establish server capacity"
    )
    fig.savefig(output / "cpu-comparison.svg")
    fig.savefig(output / "cpu-comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/results/benchmark"))
    args = parser.parse_args()
    rows = summarize(args.inputs)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "| Model | Rate | Burst | Policy | Reps | Success rps (min–max) | p50 / p95 / p99 ms |"
        " Unsuccessful % | Client-invalid runs | Target met every run |",
        "| --- | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        quantiles = " / ".join(
            f"{row[k]:.2f}" if row[k] is not None else "n/a"
            for k in ("p50_ms_mean", "p95_ms_mean", "p99_ms_mean")
        )
        lines.append(
            f"| {row['adapter']} | {row['rate']:g} | {row['burst']} | {row['policy']} | "
            f"{row['repetitions']} | {row['successful_rps_mean']:.1f} "
            f"({row['successful_rps_min']:.1f}–{row['successful_rps_max']:.1f}) | {quantiles} | "
            f"{row['unsuccessful_percent']:.2f} | {row['invalid_client_runs']} | "
            f"{row['target_met_all_runs']} |"
        )
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    charts(rows, args.output)


if __name__ == "__main__":
    main()
