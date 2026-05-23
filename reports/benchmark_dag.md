# DAG Benchmark Report

Generated at: `2026-05-23T10:08:52.896193+00:00`

Scope: generic `AgentPipeline` scheduler only. This report compares serial
subtask execution with Kahn topological layers plus `asyncio.gather`.
It is not a benchmark of the Pilot approval workflow, which intentionally
keeps business steps sequential inside each layer for approval/resume
consistency.

## Config

| Item | Value |
| --- | ---: |
| Scenarios | 50 |
| Rounds per scenario | 1 |
| Seed | 20260523 |
| Time scale | 1.0 |

## Summary

| Metric | Value |
| --- | ---: |
| Avg serial latency | 356.28 ms |
| Avg DAG latency | 137.89 ms |
| Overall speedup | 2.58x |
| Overall latency reduction | 61.30% |
| Mean scenario reduction | 61.29% |
| P50 reduction | 61.96% |
| P90 reduction | 64.64% |
| Min reduction | 51.88% |
| Max reduction | 66.59% |
| Avg layer count | 4.00 |
| Avg max layer width | 4.00 |

## Scenario Details

| Scenario | Tasks | Layer shape | Serial ms | DAG ms | Speedup | Reduction |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| office_task_001 | 9 | [2, 4, 2, 1] | 347.57 | 123.72 | 2.81x | 64.40% |
| office_task_002 | 9 | [2, 4, 2, 1] | 343.03 | 123.08 | 2.79x | 64.12% |
| office_task_003 | 9 | [2, 4, 2, 1] | 343.71 | 140.90 | 2.44x | 59.01% |
| office_task_004 | 9 | [2, 4, 2, 1] | 388.39 | 156.43 | 2.48x | 59.72% |
| office_task_005 | 9 | [2, 4, 2, 1] | 360.78 | 138.15 | 2.61x | 61.71% |
| office_task_006 | 9 | [2, 4, 2, 1] | 374.97 | 138.13 | 2.71x | 63.16% |
| office_task_007 | 9 | [2, 4, 2, 1] | 340.62 | 123.77 | 2.75x | 63.66% |
| office_task_008 | 9 | [2, 4, 2, 1] | 359.11 | 125.59 | 2.86x | 65.03% |
| office_task_009 | 9 | [2, 4, 2, 1] | 354.95 | 125.03 | 2.84x | 64.78% |
| office_task_010 | 9 | [2, 4, 2, 1] | 324.19 | 153.95 | 2.11x | 52.51% |
| office_task_011 | 9 | [2, 4, 2, 1] | 360.01 | 142.13 | 2.53x | 60.52% |
| office_task_012 | 9 | [2, 4, 2, 1] | 372.59 | 139.64 | 2.67x | 62.52% |
| office_task_013 | 9 | [2, 4, 2, 1] | 372.30 | 124.40 | 2.99x | 66.59% |
| office_task_014 | 9 | [2, 4, 2, 1] | 357.10 | 122.93 | 2.90x | 65.57% |
| office_task_015 | 9 | [2, 4, 2, 1] | 358.39 | 126.71 | 2.83x | 64.64% |
| office_task_016 | 9 | [2, 4, 2, 1] | 294.09 | 107.51 | 2.74x | 63.44% |
| office_task_017 | 9 | [2, 4, 2, 1] | 372.16 | 171.52 | 2.17x | 53.91% |
| office_task_018 | 9 | [2, 4, 2, 1] | 373.18 | 154.84 | 2.41x | 58.51% |
| office_task_019 | 9 | [2, 4, 2, 1] | 342.87 | 124.81 | 2.75x | 63.60% |
| office_task_020 | 9 | [2, 4, 2, 1] | 356.90 | 140.24 | 2.54x | 60.71% |
| office_task_021 | 9 | [2, 4, 2, 1] | 341.13 | 125.67 | 2.71x | 63.16% |
| office_task_022 | 9 | [2, 4, 2, 1] | 341.54 | 138.19 | 2.47x | 59.54% |
| office_task_023 | 9 | [2, 4, 2, 1] | 358.07 | 154.39 | 2.32x | 56.88% |
| office_task_024 | 9 | [2, 4, 2, 1] | 357.57 | 141.54 | 2.53x | 60.42% |
| office_task_025 | 9 | [2, 4, 2, 1] | 358.73 | 140.86 | 2.55x | 60.73% |
| office_task_026 | 9 | [2, 4, 2, 1] | 342.39 | 123.24 | 2.78x | 64.01% |
| office_task_027 | 9 | [2, 4, 2, 1] | 391.61 | 140.53 | 2.79x | 64.11% |
| office_task_028 | 9 | [2, 4, 2, 1] | 370.46 | 139.87 | 2.65x | 62.25% |
| office_task_029 | 9 | [2, 4, 2, 1] | 347.31 | 139.61 | 2.49x | 59.80% |
| office_task_030 | 9 | [2, 4, 2, 1] | 356.44 | 155.30 | 2.29x | 56.43% |
| office_task_031 | 9 | [2, 4, 2, 1] | 342.39 | 123.96 | 2.76x | 63.80% |
| office_task_032 | 9 | [2, 4, 2, 1] | 369.85 | 141.28 | 2.62x | 61.80% |
| office_task_033 | 9 | [2, 4, 2, 1] | 328.44 | 124.92 | 2.63x | 61.97% |
| office_task_034 | 9 | [2, 4, 2, 1] | 340.92 | 124.55 | 2.74x | 63.46% |
| office_task_035 | 9 | [2, 4, 2, 1] | 372.14 | 139.56 | 2.67x | 62.50% |
| office_task_036 | 9 | [2, 4, 2, 1] | 372.33 | 139.89 | 2.66x | 62.43% |
| office_task_037 | 9 | [2, 4, 2, 1] | 370.98 | 156.29 | 2.37x | 57.87% |
| office_task_038 | 9 | [2, 4, 2, 1] | 356.96 | 139.80 | 2.55x | 60.84% |
| office_task_039 | 9 | [2, 4, 2, 1] | 343.68 | 142.65 | 2.41x | 58.49% |
| office_task_040 | 9 | [2, 4, 2, 1] | 356.22 | 171.40 | 2.08x | 51.88% |
| office_task_041 | 9 | [2, 4, 2, 1] | 354.79 | 125.65 | 2.82x | 64.59% |
| office_task_042 | 9 | [2, 4, 2, 1] | 355.39 | 141.51 | 2.51x | 60.18% |
| office_task_043 | 9 | [2, 4, 2, 1] | 357.06 | 142.05 | 2.51x | 60.22% |
| office_task_044 | 9 | [2, 4, 2, 1] | 388.70 | 138.79 | 2.80x | 64.29% |
| office_task_045 | 9 | [2, 4, 2, 1] | 370.86 | 138.06 | 2.69x | 62.77% |
| office_task_046 | 9 | [2, 4, 2, 1] | 358.80 | 125.64 | 2.86x | 64.98% |
| office_task_047 | 9 | [2, 4, 2, 1] | 325.11 | 123.67 | 2.63x | 61.96% |
| office_task_048 | 9 | [2, 4, 2, 1] | 356.44 | 155.23 | 2.30x | 56.45% |
| office_task_049 | 9 | [2, 4, 2, 1] | 374.81 | 155.68 | 2.41x | 58.46% |
| office_task_050 | 9 | [2, 4, 2, 1] | 355.80 | 141.14 | 2.52x | 60.33% |

## Interview Wording

- I benchmarked the generic AgentPipeline scheduler, where independent DAG
  layers are executed with asyncio.gather.
- Pilot's Feishu approval workflow is not included in this latency number.
  Pilot currently advances business steps sequentially to keep approvals,
  artifacts, and resume behavior deterministic.
- The benchmark uses deterministic simulated I/O tasks, so it proves the
  scheduling strategy and reproduces the reduction number; it is not a
  production SLA or a live Feishu end-to-end result.
