"""Obsidian Markdown 持久化记忆。

将记忆条目写入 Obsidian Vault 兼容的 Markdown 文件，
支持按标签、关键词、日期范围检索。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from agent_hub.core.models import MemoryEntry


def _safe_filename(name: str) -> str:
    """将字符串转为安全的文件名片段（仅保留字母数字和连字符）。"""
    return re.sub(r"[^a-zA-Z0-9\-]", "", name)


def _parse_frontmatter(text: str) -> dict[str, object]:
    """从 Markdown 文本中解析 YAML frontmatter。

    Returns:
        frontmatter 字典；解析失败时返回空字典。
    """
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = yaml.safe_load(match.group(1))
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError:
        return {}


def _parse_body(text: str) -> str:
    """从 Markdown 文本中提取 frontmatter 之后的正文。"""
    match = re.match(r"^---\n.*?\n---\n?(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _entry_to_markdown(entry: MemoryEntry) -> str:
    """将 MemoryEntry 序列化为 Obsidian Markdown 文本（含 frontmatter）。"""
    frontmatter: dict[str, object] = {
        "memory_id": entry.memory_id,
        "created_at": entry.created_at.isoformat(),
        "tags": entry.tags,
        "session_id": entry.session_id,
        "memory_type": entry.memory_type,
        "ttl": entry.ttl,
        "is_deleted": entry.is_deleted,
    }
    fm_str = yaml.safe_dump(
        frontmatter, allow_unicode=True, default_flow_style=False, sort_keys=False
    )
    return f"---\n{fm_str}---\n\n{entry.content}\n"


def _markdown_to_entry(text: str) -> MemoryEntry | None:
    """从 Markdown 文本反序列化为 MemoryEntry。

    Returns:
        解析成功返回 MemoryEntry；失败返回 ``None``。
    """
    fm = _parse_frontmatter(text)
    if not fm:
        return None

    created_at_raw = fm.get("created_at", "")
    try:
        created_at = datetime.fromisoformat(str(created_at_raw))
    except (ValueError, TypeError):
        created_at = datetime.now()

    tags_raw = fm.get("tags")
    tags: list[str] = tags_raw if isinstance(tags_raw, list) else []

    return MemoryEntry(
        memory_id=str(fm.get("memory_id", "")),
        content=_parse_body(text),
        memory_type=str(fm.get("memory_type", "long_term")),
        session_id=fm.get("session_id"),  # type: ignore[arg-type]
        tags=tags,
        created_at=created_at,
        ttl=int(fm.get("ttl", -1)),
        is_deleted=bool(fm.get("is_deleted", False)),
    )


class PersistentMemory:
    """Obsidian Vault 持久化记忆管理器。

    文件存储路径格式::

        {vault_path}/{user_id}/{YYYY-MM}/{DD}-{session_id_short}.md

    Args:
        vault_path: Obsidian Vault 根目录，默认 ``"./data/obsidian"``。
    """

    def __init__(self, vault_path: str = "./data/obsidian") -> None:
        self._vault_path = Path(vault_path)

    def _user_dir(self, user_id: str) -> Path:
        return self._vault_path / user_id

    def save(self, entry: MemoryEntry, user_id: str) -> Path:
        """保存一条记忆到 Obsidian Vault。

        自动创建目录结构，文件名格式：``{DD}-{session_id[:8]}.md``。

        Args:
            entry: 记忆条目。
            user_id: 用户唯一标识。

        Returns:
            写入的文件路径。
        """
        dt = entry.created_at
        month_dir = self._user_dir(user_id) / f"{dt:%Y-%m}"
        month_dir.mkdir(parents=True, exist_ok=True)

        session_part = _safe_filename(entry.session_id or "nosess")[:8]
        ts_part = f"{dt:%H%M%S}"
        filename = f"{dt:%d}-{session_part}-{ts_part}.md"
        file_path = month_dir / filename

        # 如果文件名冲突，追加序号
        counter = 1
        while file_path.exists():
            filename = f"{dt:%d}-{session_part}-{ts_part}-{counter}.md"
            file_path = month_dir / filename
            counter += 1

        file_path.write_text(_entry_to_markdown(entry), encoding="utf-8")
        return file_path

    def save_batch(
        self, entries: list[MemoryEntry], user_id: str
    ) -> list[Path]:
        """批量保存多条记忆。

        Args:
            entries: 记忆条目列表。
            user_id: 用户唯一标识。

        Returns:
            写入的文件路径列表。
        """
        return [self.save(entry, user_id) for entry in entries]

    def search_by_tag(
        self, user_id: str, tag: str
    ) -> list[MemoryEntry]:
        """按标签搜索记忆。

        遍历用户目录下所有 ``.md`` 文件，返回 frontmatter ``tags``
        中包含指定标签的条目。

        Args:
            user_id: 用户唯一标识。
            tag: 要搜索的标签。

        Returns:
            匹配的记忆列表，按创建时间正序。
        """
        results: list[MemoryEntry] = []
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return results

        for md_file in sorted(user_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            tags = fm.get("tags", [])
            if isinstance(tags, list) and tag in tags:
                entry = _markdown_to_entry(text)
                if entry is not None and not entry.is_deleted:
                    results.append(entry)
        return results

    def search_by_keyword(
        self, user_id: str, keyword: str, limit: int = 10
    ) -> list[MemoryEntry]:
        """按关键词搜索记忆（正文 + frontmatter 模糊匹配）。

        Args:
            user_id: 用户唯一标识。
            keyword: 搜索关键词（支持正则）。
            limit: 最大返回条数，默认 10。

        Returns:
            匹配的记忆列表，按创建时间正序。
        """
        results: list[MemoryEntry] = []
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return results

        pattern = re.compile(keyword, re.IGNORECASE)
        for md_file in sorted(user_dir.rglob("*.md")):
            if len(results) >= limit:
                break
            text = md_file.read_text(encoding="utf-8")
            if pattern.search(text):
                entry = _markdown_to_entry(text)
                if entry is not None and not entry.is_deleted:
                    results.append(entry)
        return results

    def load_recent(
        self, user_id: str, days: int = 7
    ) -> list[MemoryEntry]:
        """加载最近 N 天的所有记忆。

        通过目录名 ``{YYYY-MM}`` 和文件名前两位 ``{DD}`` 过滤日期。

        Args:
            user_id: 用户唯一标识。
            days: 回溯天数，默认 7。

        Returns:
            匹配的记忆列表，按创建时间正序。
        """
        results: list[MemoryEntry] = []
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return results

        cutoff = datetime.now() - timedelta(days=days)

        for md_file in sorted(user_dir.rglob("*.md")):
            text = md_file.read_text(encoding="utf-8")
            entry = _markdown_to_entry(text)
            if entry is not None and entry.created_at >= cutoff and not entry.is_deleted:
                results.append(entry)
        return results

    def soft_delete(self, memory_id: str, user_id: str) -> bool:
        """软删除一条记忆（标记 is_deleted=True 而非物理删除）。

        Args:
            memory_id: 记忆唯一标识。
            user_id: 用户唯一标识。

        Returns:
            是否成功找到并标记。
        """
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return False

        for md_file in user_dir.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            if fm.get("memory_id") == memory_id:
                entry = _markdown_to_entry(text)
                if entry is not None:
                    entry.is_deleted = True
                    md_file.write_text(_entry_to_markdown(entry), encoding="utf-8")
                    return True
        return False

    def get_by_id(self, memory_id: str, user_id: str) -> MemoryEntry | None:
        """按 memory_id 查找记忆。

        Args:
            memory_id: 记忆唯一标识。
            user_id: 用户唯一标识。

        Returns:
            匹配的 MemoryEntry 或 None。
        """
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return None

        for md_file in user_dir.rglob("*.md"):
            text = md_file.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            if fm.get("memory_id") == memory_id:
                return _markdown_to_entry(text)
        return None

    def get_context_for_llm(
        self, user_id: str, max_tokens: int = 2000
    ) -> str:
        """将最近的持久记忆压缩为上下文字符串。

        按时间倒序取最新条目，用 ``len(text) // 4`` 近似 token 数。

        Args:
            user_id: 用户唯一标识。
            max_tokens: 允许的最大 token 数，默认 2000。

        Returns:
            拼接后的上下文字符串。
        """
        entries = self.load_recent(user_id, days=7)
        if not entries:
            return ""

        lines: list[str] = []
        used_chars = 0
        max_chars = max_tokens * 4

        # 从最新开始向前追溯
        for entry in reversed(entries):
            tag_str = ", ".join(entry.tags) if entry.tags else ""
            header = f"[{entry.created_at:%Y-%m-%d %H:%M}]"
            if tag_str:
                header += f" #{tag_str}"
            line = f"{header}\n{entry.content}"
            line_chars = len(line)
            if used_chars + line_chars > max_chars:
                break
            lines.append(line)
            used_chars += line_chars

        lines.reverse()
        return "\n\n".join(lines)
