"""演示前自检脚本：在飞书 + 真实链路启用前快速发现配置错误。

聚焦“现场会失败”的硬约束，避免上场后只能靠日志定位：

1. ``python-pptx`` 是否可导入（``real.slide.render_pptx`` 必须的真实渲染依赖）；
2. 当前 ``Settings`` 加载的关键字段（``PILOT_USE_REAL_CHAIN`` /
   ``FEISHU_DEFAULT_FOLDER_TOKEN`` / ``FEISHU_ADMIN_OPEN_ID`` /
   ``FEISHU_APP_ID`` / ``FEISHU_APP_SECRET`` / ``FEISHU_BOT_OPEN_ID``
   等）是否符合“真实演示档位”；
3. ``pilot_artifact_dir`` 是否存在且可写。

退出码：
    0 — 全部通过
    1 — 至少一项失败（见 stderr 输出）

用法：
    python scripts/demo_preflight.py
    python scripts/demo_preflight.py --strict-real-chain   # 同时校验真实链路必备字段
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_hub.config.settings import get_settings  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✘{RESET} {msg}")


def check_python_pptx(failures: list[str]) -> None:
    print("\n[1/3] 检查 python-pptx（真实 PPTX 渲染依赖）")
    try:
        import pptx  # noqa: F401
    except ImportError as exc:
        _fail(
            "未安装 python-pptx；real.slide.render_pptx 默认会抛错。"
            "请运行：pip install python-pptx",
        )
        failures.append(f"python-pptx 导入失败：{exc}")
        return
    _ok(f"python-pptx 可用（version={pptx.__version__})")
    if os.environ.get("PILOT_ALLOW_PPTX_STUB", "").lower() in ("1", "true", "yes"):
        _warn(
            "PILOT_ALLOW_PPTX_STUB=true 已启用：缺依赖时会降级为占位 stub，"
            "演示前请确认这是预期。",
        )


def check_artifact_dir(failures: list[str], artifact_dir: str) -> None:
    print(f"\n[2/3] 检查 artifact 目录（pilot_artifact_dir={artifact_dir}）")
    path = Path(artifact_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _fail(f"无法创建 artifact 目录 {path}: {exc}")
        failures.append(f"artifact_dir 不可创建：{exc}")
        return
    # 真正写一个临时文件验证可写
    try:
        with tempfile.NamedTemporaryFile(
            dir=path, prefix=".preflight_", suffix=".tmp", delete=True,
        ):
            pass
    except OSError as exc:
        _fail(f"artifact 目录 {path} 不可写：{exc}")
        failures.append(f"artifact_dir 不可写：{exc}")
        return
    _ok(f"artifact 目录可写：{path.resolve()}")


def check_feishu_and_real_chain(
    failures: list[str], strict_real_chain: bool
) -> None:
    print("\n[3/3] 检查 Settings 中的飞书与真实链路口径")
    s = get_settings()

    _ok(f"PILOT_ENABLED={s.pilot_enabled}")
    _ok(f"PILOT_USE_REAL_CHAIN={s.pilot_use_real_chain}")
    _ok(f"FEISHU_ENABLED={s.feishu_enabled}")
    _ok(f"FEISHU_USE_LONG_CONN={s.feishu_use_long_conn}")

    require_real = strict_real_chain or s.pilot_use_real_chain
    if not require_real:
        _warn(
            "未启用真实链路（PILOT_USE_REAL_CHAIN=false），跳过真实凭据校验；"
            "如需现场演示真实 PPTX/Drive，请设置为 true 后重跑。",
        )
        return

    if not s.feishu_enabled:
        _fail("启用了真实链路但 FEISHU_ENABLED=false，无法走飞书产物链。")
        failures.append("FEISHU_ENABLED 必须为 true")
        return

    required_pairs: list[tuple[str, str]] = [
        ("FEISHU_APP_ID", s.feishu_app_id),
        ("FEISHU_APP_SECRET", s.feishu_app_secret),
        ("FEISHU_BOT_OPEN_ID", s.feishu_bot_open_id),
        ("FEISHU_DEFAULT_FOLDER_TOKEN", s.feishu_default_folder_token),
        ("FEISHU_ADMIN_OPEN_ID", s.feishu_admin_open_id),
    ]
    for name, value in required_pairs:
        if not value:
            _fail(f"{name} 为空；真实链路上传/分享/审批通知都会失败。")
            failures.append(f"缺少 {name}")
        else:
            masked = value[:4] + "…" if len(value) > 6 else value
            _ok(f"{name}={masked}")

    # 长连接以外的部署：webhook 必须配 verification_token
    if not s.feishu_use_long_conn and not s.feishu_verification_token:
        _fail(
            "FEISHU_USE_LONG_CONN=false 且未配置 FEISHU_VERIFICATION_TOKEN，"
            "webhook 验签会失败。",
        )
        failures.append("缺少 FEISHU_VERIFICATION_TOKEN（webhook 模式）")


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent-Pilot 演示前自检")
    parser.add_argument(
        "--strict-real-chain",
        action="store_true",
        help="即使当前 PILOT_USE_REAL_CHAIN=false，也按真实链路口径校验飞书凭据",
    )
    args = parser.parse_args()

    failures: list[str] = []
    settings = get_settings()
    check_python_pptx(failures)
    check_artifact_dir(failures, settings.pilot_artifact_dir)
    check_feishu_and_real_chain(failures, args.strict_real_chain)

    print()
    if failures:
        print(f"{RED}演示前自检未通过，共 {len(failures)} 项需修复：{RESET}")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"{GREEN}演示前自检全部通过，可以现场演示。{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
