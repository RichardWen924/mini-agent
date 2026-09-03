from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Protocol

from .models import Message, Role


@dataclass(frozen=True)
class ToolCall:
    """表示一次待执行的工具调用请求。"""

    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ProviderResponse:
    """表示模型的一次响应，可以是普通文本或工具调用。"""

    message: Message
    tool_call: ToolCall | None = None


class Provider(Protocol):
    """定义模型适配器必须提供的接口。"""

    def complete(
        self,
        messages: Sequence[Message],
    ) -> ProviderResponse:
        """根据消息列表生成一次模型响应。"""


def _parse_number(value: str) -> float | None:
    """将文本解析成有限浮点数，解析失败时返回 None。"""

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None

    return number


def _parse_calculation_command(
    content: str,
) -> ToolCall | None:
    """将 calc 命令解析为 calculator 工具调用。"""

    parts = content.split()

    if len(parts) != 4:
        return None

    if parts[0].lower() != "calc":
        return None

    operation = parts[1].lower()
    a = _parse_number(parts[2])
    b = _parse_number(parts[3])

    if a is None or b is None:
        return None

    return ToolCall(
        name="calculator",
        arguments={
            "operation": operation,
            "a": a,
            "b": b,
        },
    )


class RuleBasedProvider:
    """用于本地开发和测试的确定性模型适配器。"""

    def complete(
        self,
        messages: Sequence[Message],
    ) -> ProviderResponse:
        """根据最后一条消息生成固定规则响应。"""

        if not messages:
            return ProviderResponse(
                message=Message(
                    Role.ASSISTANT,
                    "请先提供一条消息。",
                )
            )

        last_message = messages[-1]

        if last_message.role == Role.TOOL:
            return ProviderResponse(
                message=Message(
                    Role.ASSISTANT,
                    f"工具结果：{last_message.content}",
                )
            )

        if last_message.role != Role.USER:
            return ProviderResponse(
                message=Message(
                    Role.ASSISTANT,
                    "请提供用户问题。",
                )
            )

        content = last_message.content.strip()

        if content.lower() == "hello":
            return ProviderResponse(
                message=Message(
                    Role.ASSISTANT,
                    "Hello!",
                )
            )

        tool_call = _parse_calculation_command(content)

        if tool_call is not None:
            return ProviderResponse(
                message=Message(
                    Role.ASSISTANT,
                    "我将调用计算器。",
                ),
                tool_call=tool_call,
            )

        return ProviderResponse(
            message=Message(
                Role.ASSISTANT,
                f"你说的是：{content}",
            )
        )
