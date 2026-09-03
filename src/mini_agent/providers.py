import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI


InputItem = Mapping[str, object]
ToolDefinition = Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    """表示模型请求执行的一次工具调用。"""

    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ProviderResponse:
    """表示模型的一次文本响应和可选工具调用。"""

    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    output_items: tuple[dict[str, object], ...] = ()


class Provider(Protocol):
    """定义模型适配器必须提供的接口。"""

    def complete(
        self,
        input_items: Sequence[InputItem],
        tool_definitions: Sequence[ToolDefinition],
    ) -> ProviderResponse:
        """根据输入和工具定义生成一次模型响应。"""


class OpenAIProvider:
    """使用 OpenAI Responses API 生成文本或工具调用。"""

    def __init__(
        self,
        model: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        """创建 Provider，并允许测试注入 fake client。"""

        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        is_deepseek = bool(deepseek_key) or os.getenv("MINI_AGENT_PROVIDER") == "deepseek"
        if is_deepseek:
            default_model = "deepseek-v4-flash"
            api_key = deepseek_key or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
        else:
            default_model = "gpt-5.6-sol"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        self.model = model or os.getenv("OPENAI_MODEL", default_model)

        if client is not None:
            self.client = client
            return

        if not api_key:
            raise ValueError("OPENAI_API_KEY or DEEPSEEK_API_KEY is not set")

        client_options: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        self.client = OpenAI(**client_options)

    def complete(
        self,
        input_items: Sequence[InputItem],
        tool_definitions: Sequence[ToolDefinition],
    ) -> ProviderResponse:
        """调用 Responses API，并解析文本和 function_call。"""

        if not input_items:
            raise ValueError("input_items must not be empty")

        response = self.client.responses.create(
            model=self.model,
            input=[dict(item) for item in input_items],
            tools=[dict(tool) for tool in tool_definitions],
        )

        output_items = tuple(
            self._serialize_output_item(item)
            for item in getattr(response, "output", [])
        )
        tool_calls = tuple(
            self._parse_tool_call(item)
            for item in output_items
            if item.get("type") == "function_call"
        )

        return ProviderResponse(
            text=getattr(response, "output_text", "") or "",
            tool_calls=tool_calls,
            output_items=output_items,
        )

    @staticmethod
    def _serialize_output_item(item: object) -> dict[str, object]:
        """把 SDK 响应对象转成可再次提交给 API 的字典。"""

        if isinstance(item, Mapping):
            return dict(item)

        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return dict(model_dump(exclude_none=True))

        return dict(vars(item))

    @staticmethod
    def _parse_tool_call(item: Mapping[str, object]) -> ToolCall:
        """解析一个 function_call 输出项。"""

        call_id = item.get("call_id")
        name = item.get("name")
        raw_arguments = item.get("arguments", "{}")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise ValueError("function_call is missing call_id or name")

        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("function_call arguments are not valid JSON") from exc
        else:
            arguments = raw_arguments

        if not isinstance(arguments, dict):
            raise ValueError("function_call arguments must be a JSON object")

        return ToolCall(call_id, name, arguments)
