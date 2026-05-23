#!/usr/bin/env python3
"""Benchmark the generic AgentPipeline DAG scheduler.

This script is intentionally scoped to the generic core pipeline scheduler:
Kahn-style topological layers + asyncio.gather for each independent layer.

It does not benchmark the Pilot business execution flow. Pilot keeps step
execution sequential inside each layer so approval, artifact, and resume state
remain easy to reason about.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from agent_hub.core.models import SubTask
from agent_hub.core.pipeline import _topological_layers


@dataclass(frozen=True)
class Scenario:
    name: str
    tasks: list[SubTask]
    durations_s: dict[str, float]


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    task_count: int
    layer_count: int
    max_layer_width: int
    expected_serial_ms: float
    expected_dag_ms: float
    measured_serial_ms: float
    measured_dag_ms: float
    speedup: float
    reduction_pct: float
    layer_shape: list[int]


async def _mock_io_work(duration_s: float) -> None:
    await asyncio.sleep(duration_s)


def _task(
    scenario_id: str,
    local_id: str,
    description: str,
    agent: str,
    deps: list[str] | None = None,
) -> SubTask:
    deps = deps or []
    return SubTask(
        subtask_id=f"{scenario_id}.{local_id}",
        description=description,
        required_agents=[agent],
        priority=1,
        depends_on=[f"{scenario_id}.{dep}" for dep in deps],
    )


def _duration(
    rng: random.Random,
    base_ms: float,
    jitter_ms: float,
    time_scale: float,
) -> float:
    ms = max(1.0, base_ms + rng.uniform(-jitter_ms, jitter_ms))
    return round(ms * time_scale / 1000, 4)


def build_office_scenario(
    index: int,
    rng: random.Random,
    *,
    time_scale: float,
) -> Scenario:
    """Build a deterministic office-like DAG with useful parallel width."""
    scenario_id = f"case_{index:03d}"

    task_defs = [
        (
            "context",
            "Read IM context and attachments",
            "retrieval_agent",
            [],
            24,
            5,
        ),
        ("policy", "Check requester policy", "tool_agent", [], 14, 4),
        (
            "kb_search",
            "Search internal knowledge snippets",
            "retrieval_agent",
            ["context"],
            42,
            9,
        ),
        (
            "metric_fetch",
            "Fetch structured business metrics",
            "tool_agent",
            ["context"],
            38,
            8,
        ),
        (
            "file_parse",
            "Parse uploaded notes or sheets",
            "tool_agent",
            ["context"],
            35,
            8,
        ),
        (
            "owner_lookup",
            "Lookup owners and approval hints",
            "retrieval_agent",
            ["context"],
            28,
            7,
        ),
        (
            "brief_draft",
            "Draft task brief from retrieved context",
            "llm_agent",
            ["kb_search", "file_parse"],
            44,
            9,
        ),
        (
            "risk_summary",
            "Summarize risk and dependency notes",
            "llm_agent",
            ["metric_fetch", "owner_lookup", "policy"],
            36,
            8,
        ),
        (
            "final_merge",
            "Merge final response for the user",
            "llm_agent",
            ["brief_draft", "risk_summary"],
            24,
            5,
        ),
    ]

    tasks: list[SubTask] = []
    durations: dict[str, float] = {}
    for local_id, desc, agent, deps, base_ms, jitter_ms in task_defs:
        task = _task(scenario_id, local_id, desc, agent, deps)
        tasks.append(task)
        durations[task.subtask_id] = _duration(
            rng,
            base_ms,
            jitter_ms,
            time_scale,
        )

    return Scenario(
        name=f"office_task_{index:03d}",
        tasks=tasks,
        durations_s=durations,
    )


async def run_serial(scenario: Scenario) -> float:
    start = time.perf_counter()
    for task in scenario.tasks:
        await _mock_io_work(scenario.durations_s[task.subtask_id])
    return (time.perf_counter() - start) * 1000


async def run_dag_parallel(scenario: Scenario) -> float:
    layers = _topological_layers(scenario.tasks)
    start = time.perf_counter()
    for layer in layers:
        await asyncio.gather(*(
            _mock_io_work(scenario.durations_s[task.subtask_id])
            for task in layer
        ))
    return (time.perf_counter() - start) * 1000


def expected_times(scenario: Scenario) -> tuple[float, float, list[int]]:
    layers = _topological_layers(scenario.tasks)
    serial_ms = sum(scenario.durations_s.values()) * 1000
    dag_ms = sum(
        max(scenario.durations_s[task.subtask_id] for task in layer) * 1000
        for layer in layers
    )
    return serial_ms, dag_ms, [len(layer) for layer in layers]


async def run_scenario(scenario: Scenario, rounds: int) -> ScenarioResult:
    serial_runs: list[float] = []
    dag_runs: list[float] = []

    for _ in range(rounds):
        serial_runs.append(await run_serial(scenario))
        dag_runs.append(await run_dag_parallel(scenario))

    expected_serial, expected_dag, layer_shape = expected_times(scenario)
    measured_serial = statistics.mean(serial_runs)
    measured_dag = statistics.mean(dag_runs)
    speedup = measured_serial / measured_dag if measured_dag > 0 else 0.0
    reduction = (
        (1 - measured_dag / measured_serial) * 100
        if measured_serial > 0
        else 0.0
    )

    return ScenarioResult(
        name=scenario.name,
        task_count=len(scenario.tasks),
        layer_count=len(layer_shape),
        max_layer_width=max(layer_shape) if layer_shape else 0,
        expected_serial_ms=round(expected_serial, 2),
        expected_dag_ms=round(expected_dag, 2),
        measured_serial_ms=round(measured_serial, 2),
        measured_dag_ms=round(measured_dag, 2),
        speedup=round(speedup, 3),
        reduction_pct=round(reduction, 2),
        layer_shape=layer_shape,
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * pct)
    return sorted_values[idx]


def aggregate(results: list[ScenarioResult]) -> dict[str, Any]:
    serial_values = [r.measured_serial_ms for r in results]
    dag_values = [r.measured_dag_ms for r in results]
    reduction_values = [r.reduction_pct for r in results]

    avg_serial = statistics.mean(serial_values) if serial_values else 0.0
    avg_dag = statistics.mean(dag_values) if dag_values else 0.0
    overall_reduction = (
        (1 - avg_dag / avg_serial) * 100
        if avg_serial > 0
        else 0.0
    )

    return {
        "scenario_count": len(results),
        "avg_serial_ms": round(avg_serial, 2),
        "avg_dag_ms": round(avg_dag, 2),
        "overall_speedup": round(avg_serial / avg_dag, 3)
        if avg_dag > 0
        else 0.0,
        "overall_reduction_pct": round(overall_reduction, 2),
        "mean_scenario_reduction_pct": round(
            statistics.mean(reduction_values), 2,
        )
        if reduction_values
        else 0.0,
        "p50_reduction_pct": round(_percentile(reduction_values, 0.50), 2),
        "p90_reduction_pct": round(_percentile(reduction_values, 0.90), 2),
        "min_reduction_pct": round(min(reduction_values), 2)
        if reduction_values
        else 0.0,
        "max_reduction_pct": round(max(reduction_values), 2)
        if reduction_values
        else 0.0,
        "avg_layer_count": round(
            statistics.mean(r.layer_count for r in results), 2,
        )
        if results
        else 0.0,
        "avg_max_layer_width": round(
            statistics.mean(r.max_layer_width for r in results), 2,
        )
        if results
        else 0.0,
    }


def write_reports(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    results: list[ScenarioResult],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_dag.json"
    md_path = output_dir / "benchmark_dag.md"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "generic AgentPipeline DAG scheduler, not Pilot approval flow",
        "config": {
            "scenarios": args.scenarios,
            "rounds": args.rounds,
            "seed": args.seed,
            "time_scale": args.time_scale,
        },
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    detail_rows = "\n".join(
        "| {name} | {task_count} | {layer_shape} | {measured_serial_ms:.2f} | "
        "{measured_dag_ms:.2f} | {speedup:.2f}x | {reduction_pct:.2f}% |".format(
            **asdict(result),
        )
        for result in results
    )
    md = f"""# DAG Benchmark Report

Generated at: `{payload["generated_at"]}`

Scope: generic `AgentPipeline` scheduler only. This report compares serial
subtask execution with Kahn topological layers plus `asyncio.gather`.
It is not a benchmark of the Pilot approval workflow, which intentionally
keeps business steps sequential inside each layer for approval/resume
consistency.

## Config

| Item | Value |
| --- | ---: |
| Scenarios | {args.scenarios} |
| Rounds per scenario | {args.rounds} |
| Seed | {args.seed} |
| Time scale | {args.time_scale} |

## Summary

| Metric | Value |
| --- | ---: |
| Avg serial latency | {summary["avg_serial_ms"]:.2f} ms |
| Avg DAG latency | {summary["avg_dag_ms"]:.2f} ms |
| Overall speedup | {summary["overall_speedup"]:.2f}x |
| Overall latency reduction | {summary["overall_reduction_pct"]:.2f}% |
| Mean scenario reduction | {summary["mean_scenario_reduction_pct"]:.2f}% |
| P50 reduction | {summary["p50_reduction_pct"]:.2f}% |
| P90 reduction | {summary["p90_reduction_pct"]:.2f}% |
| Min reduction | {summary["min_reduction_pct"]:.2f}% |
| Max reduction | {summary["max_reduction_pct"]:.2f}% |
| Avg layer count | {summary["avg_layer_count"]:.2f} |
| Avg max layer width | {summary["avg_max_layer_width"]:.2f} |

## Scenario Details

| Scenario | Tasks | Layer shape | Serial ms | DAG ms | Speedup | Reduction |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
{detail_rows}

## Interview Wording

- I benchmarked the generic AgentPipeline scheduler, where independent DAG
  layers are executed with asyncio.gather.
- Pilot's Feishu approval workflow is not included in this latency number.
  Pilot currently advances business steps sequentially to keep approvals,
  artifacts, and resume behavior deterministic.
- The benchmark uses deterministic simulated I/O tasks, so it proves the
  scheduling strategy and reproduces the reduction number; it is not a
  production SLA or a live Feishu end-to-end result.
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    scenarios = [
        build_office_scenario(i, rng, time_scale=args.time_scale)
        for i in range(1, args.scenarios + 1)
    ]

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        results.append(await run_scenario(scenario, args.rounds))

    summary = aggregate(results)
    json_path, md_path = write_reports(
        args.output_dir,
        args=args,
        results=results,
        summary=summary,
    )

    print("DAG benchmark complete")
    print(f"  scenarios: {summary['scenario_count']}")
    print(f"  rounds: {args.rounds}")
    print(f"  avg serial: {summary['avg_serial_ms']:.2f} ms")
    print(f"  avg dag: {summary['avg_dag_ms']:.2f} ms")
    print(f"  speedup: {summary['overall_speedup']:.2f}x")
    print(f"  reduction: {summary['overall_reduction_pct']:.2f}%")
    print(f"  json: {json_path}")
    print(f"  markdown: {md_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs DAG-layered asyncio.gather execution.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=50,
        help="Number of deterministic office task DAGs to generate.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Repeated runs per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260523,
        help="Random seed for reproducible durations.",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Scale simulated I/O durations; lower values run faster.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "reports",
        help="Directory for benchmark_dag.json and benchmark_dag.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scenarios < 1:
        raise SystemExit("--scenarios must be >= 1")
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.time_scale <= 0:
        raise SystemExit("--time-scale must be > 0")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
