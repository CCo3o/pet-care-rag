"""终端展示层：整理阅读格式，保留聊天原文与工具结果用于追问和调试。"""
from __future__ import annotations

import json
from pathlib import PureWindowsPath
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.chat import ChatReply


def terminal_text(answer: str) -> str:
    """把常见的 Markdown 加粗和标题变成终端可读文本。

    只去掉成对的 **/*** 标记，保留金额、换行、编号及 118 * 7 等算式。
    原始回答不变，以便未来网页端正常渲染 Markdown。
    """
    answer = re.sub(r"(?<!\*)(\*{2,3})(?=\S)(.+?)\1(?!\*)", r"\2", answer, flags=re.DOTALL)
    return re.sub(r"(?m)^ {0,3}#{1,6}[ \t]+", "", answer)


def render_reply(reply: ChatReply, *, debug: bool = False) -> str:
    """普通模式展示回答和来源名称；开发模式额外显示原始工具记录。"""
    lines = []
    if debug:
        lines.extend("🔎 " + json.dumps(event, ensure_ascii=False) for event in reply.trace)
    lines.append(f"\n💬 {terminal_text(reply.answer)}")
    if reply.sources:
        lines.append("\n📎 检索参考来源：")
        for source in reply.sources:
            # 同时兼容 Windows 的反斜杠与正斜杠路径，普通展示不暴露磁盘目录。
            lines.append(f"   - 《{PureWindowsPath(source).stem}》")
    return "\n".join(lines)
