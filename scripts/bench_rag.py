#!/usr/bin/env python3
"""DAG 拓扑排序并行加速基准测试。

比较串行 vs asyncio.gather 并行执行子任务的端到端耗时。

用法::

    python scripts/bench_rag.py
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent_hub.core.models import SubTask
from agent_hub.core.pipeline import _topological_layers


async def _mock_subtask(duration_s: float) -> str:
    """模拟子任务执行（sleep 指定时间）。"""
    await asyncio.sleep(duration_s)
    return f"done({duration_s}s)"


async def bench_serial(task_durations: dict[str, float]) -> float:
    """串行执行所有任务，返回总耗时。"""
    start = time.monotonic()
    for dur in task_durations.values():
        await _mock_subtask(dur)
    return time.monotonic() - start


async def bench_parallel_dag(
    sub_tasks: list[SubTask],
    task_durations: dict[str, float],
) -> float:
    """DAG 拓扑排序并行执行，返回总耗时。"""
    layers = _topological_layers(sub_tasks)
    start = time.monotonic()
    for layer in layers:
        coros = [_mock_subtask(task_durations[st.subtask_id]) for st in layer]
        await asyncio.gather(*coros)
    return time.monotonic() - start


async def run_3_task_scenario() -> None:
    """3 子任务场景：task_1 和 task_2 无依赖可并行，task_3 依赖前两者。"""
    sub_tasks = [
        SubTask(
            subtask_id="task_1",
            description="独立子任务1（如：查天气）",
            required_agents=["tool_agent"],
            priority=1,
            depends_on=[],
        ),
        SubTask(
            subtask_id="task_2",
            description="独立子任务2（如：查数据库）",
            required_agents=["retrieval_agent"],
            priority=1,
            depends_on=[],
        ),
        SubTask(
            subtask_id="task_3",
            description="依赖子任务（如：汇总生成报告）",
            required_agents=["llm_agent"],
            priority=1,
            depends_on=["task_1", "task_2"],
        ),
    ]

    task_durations = {
        "task_1": 0.2,
        "task_2": 0.2,
        "task_3": 0.3,
    }

    # DAG 分层验证
    layers = _topological_layers(sub_tasks)
    print(f"  DAG 分层结果:")
    for i, layer in enumerate(layers):
        ids = [st.subtask_id for st in layer]
        print(f"    Layer {i}: {ids}")

    n_rounds = 20
    serial_times: list[float] = []
    parallel_times: list[float] = []

    for _ in range(n_rounds):
        s = await bench_serial(task_durations)
        serial_times.append(s)
        p = await bench_parallel_dag(sub_tasks, task_durations)
        parallel_times.append(p)

    avg_serial = statistics.mean(serial_times)
    avg_parallel = statistics.mean(parallel_times)
    speedup = avg_serial / avg_parallel if avg_parallel > 0 else 0
    reduction = (1 - avg_parallel / avg_serial) * 100

    print(f"\n  串行平均耗时:   {avg_serial * 1000:.0f} ms")
    print(f"  并行平均耗时:   {avg_parallel * 1000:.0f} ms")
    print(f"  加速比:         {speedup:.2f}x")
    print(f"  耗时缩短:       {reduction:.0f}%")

    print("\n  ⚠ 注意: 加速效果取决于子任务实际耗时比例")
    print("  上述模拟中每个子任务有 200-300ms 的 I/O 延迟，")
    print("  实际项目中 LLM 调用延迟可能更大，加速效果更显著。")


async def main() -> None:
    print("=" * 60)
    print("DAG 拓扑排序并行加速基准测试")
    print("=" * 60)

    print("\n场景: 3 子任务 (2 并行 + 1 依赖)")
    await run_3_task_scenario()


if __name__ == "__main__":
    asyncio.run(main())
