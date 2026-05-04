"""SkillRegistry 契约测试。"""

from __future__ import annotations

import asyncio

import pytest

from agent_hub.pilot.skills import (
    DryRunUnsupported,
    SkillInvocation,
    SkillNotFound,
    SkillRegistry,
    SkillResult,
    SkillSpec,
    register_fake_skills,
)


def _make_spec(
    name: str = "noop",
    *,
    side_effect: str = "read",
    requires_approval: bool = False,
    supports_dry_run: bool = True,
    timeout_ms: int = 5_000,
    scopes: list[str] | None = None,
) -> SkillSpec:
    return SkillSpec(
        skill_name=name,
        side_effect=side_effect,  # type: ignore[arg-type]
        requires_approval=requires_approval,
        supports_dry_run=supports_dry_run,
        timeout_ms=timeout_ms,
        scopes=scopes or [],
    )


async def _ok(inv: SkillInvocation) -> SkillResult:
    return SkillResult(skill_name=inv.skill_name, success=True, output={"x": 1})


def test_register_and_get() -> None:
    r = SkillRegistry()
    r.register(_make_spec("a"), _ok)
    assert r.get("a") is not None
    assert r.get("missing") is None


def test_register_duplicate_rejected() -> None:
    r = SkillRegistry()
    r.register(_make_spec("a"), _ok)
    with pytest.raises(ValueError):
        r.register(_make_spec("a"), _ok)
    r.register(_make_spec("a"), _ok, allow_overwrite=True)


def test_must_get_raises_for_unknown() -> None:
    r = SkillRegistry()
    with pytest.raises(SkillNotFound):
        r.must_get("ghost")


def test_list_for_scopes_filters() -> None:
    r = SkillRegistry()
    r.register(_make_spec("free"), _ok)
    r.register(_make_spec("im", scopes=["im:write"]), _ok)
    r.register(_make_spec("drive", scopes=["drive:write"]), _ok)

    visible = {s.skill_name for s in r.list_for_scopes(["im:write"])}
    assert visible == {"free", "im"}


@pytest.mark.asyncio
async def test_invoke_success_overrides_metadata() -> None:
    r = SkillRegistry()
    r.register(_make_spec("a"), _ok)
    result = await r.invoke(SkillInvocation(skill_name="a"))
    assert result.skill_name == "a"
    assert result.success is True
    assert result.duration_ms >= 0
    assert result.dry_run is False


@pytest.mark.asyncio
async def test_invoke_unknown_raises() -> None:
    r = SkillRegistry()
    with pytest.raises(SkillNotFound):
        await r.invoke(SkillInvocation(skill_name="ghost"))


@pytest.mark.asyncio
async def test_dry_run_requires_supports_dry_run_flag() -> None:
    r = SkillRegistry()
    r.register(_make_spec("a", supports_dry_run=False), _ok)
    with pytest.raises(DryRunUnsupported):
        await r.dry_run(SkillInvocation(skill_name="a"))


@pytest.mark.asyncio
async def test_dry_run_marks_invocation() -> None:
    seen: dict[str, bool] = {}

    async def capture(inv: SkillInvocation) -> SkillResult:
        seen["dry_run"] = inv.dry_run
        return SkillResult(skill_name=inv.skill_name, success=True)

    r = SkillRegistry()
    r.register(_make_spec("a"), capture)
    result = await r.dry_run(SkillInvocation(skill_name="a"))
    assert seen["dry_run"] is True
    assert result.dry_run is True


@pytest.mark.asyncio
async def test_invoke_timeout_returns_failed_result() -> None:
    async def slow(inv: SkillInvocation) -> SkillResult:
        await asyncio.sleep(1.0)
        return SkillResult(skill_name=inv.skill_name, success=True)

    r = SkillRegistry()
    r.register(_make_spec("slow", timeout_ms=50), slow)
    result = await r.invoke(SkillInvocation(skill_name="slow"))
    assert result.success is False
    assert "timeout" in (result.error or "")


@pytest.mark.asyncio
async def test_invoke_exception_returns_failed_result() -> None:
    async def boom(inv: SkillInvocation) -> SkillResult:
        raise RuntimeError("kaboom")

    r = SkillRegistry()
    r.register(_make_spec("boom"), boom)
    result = await r.invoke(SkillInvocation(skill_name="boom"))
    assert result.success is False
    assert result.error == "kaboom"


@pytest.mark.asyncio
async def test_register_fake_skills_full_set() -> None:
    r = SkillRegistry()
    register_fake_skills(r)
    names = {s.skill_name for s in r.list_skills()}
    assert {
        "fake_im_fetch",
        "fake_brief_generate",
        "fake_slidespec_generate",
        "fake_pptx_render",
        "fake_drive_upload",
        "fake_slides_share",
        "fake_im_send_summary",
    }.issubset(names)

    write_skills = {
        s.skill_name for s in r.list_skills() if s.side_effect in ("write", "share")
    }
    for name in write_skills:
        assert r.must_get(name).spec.requires_approval is True

    result = await r.invoke(SkillInvocation(
        skill_name="fake_brief_generate",
        params={"title": "Demo"},
    ))
    assert result.success is True
    assert result.artifact_payload is not None
    assert result.artifact_payload["type"] == "project_brief_md"
