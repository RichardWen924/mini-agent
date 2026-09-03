import json

from .models import Message, Role
from .prompts import build_prompt, trim_history
from .providers import Provider
from .tools import ToolError, ToolRegistry


class AgentError(Exception):
    """表示 Agent 流程执行失败。"""


class Agent:
    """协调 prompt、Provider 和工具执行的最小 Agent。"""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 3,
        max_history: int = 20,
    ) -> None:
        """创建 Agent 并配置模型、工具和循环限制。"""

        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")

        if max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")

        if max_history < 0:
            raise ValueError("max_history must not be negative")

        self.provider = provider
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_history = max_history
        self.history: list[Message] = []

    def ask(self, user_input: str) -> str:
        """处理一次用户问题并返回最终文本回复。"""

        prompt = build_prompt(
            self.system_prompt,
            self.history,
            user_input,
            self.max_history,
        )
        self.history.append(Message(Role.USER, user_input))

        for _ in range(self.max_steps):
            try:
                response = self.provider.complete(prompt)
            except Exception as exc:
                raise AgentError("model provider failed") from exc

            self.history.append(response.message)

            if response.tool_call is None:
                return response.message.content

            try:
                tool_content = self.tools.call(
                    response.tool_call.name,
                    response.tool_call.arguments,
                )
            except ToolError as exc:
                tool_content = json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                )

            self.history.append(Message(Role.TOOL, tool_content))
            prompt = self._build_history_prompt()

        return "已达到最大处理步骤，无法完成请求。"

    def _build_history_prompt(self) -> list[Message]:
        """根据系统提示词和当前历史构造后续模型请求。"""

        return [
            Message(Role.SYSTEM, self.system_prompt),
            *trim_history(self.history, self.max_history),
        ]
