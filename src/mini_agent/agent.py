from .models import Message, Role
from .prompts import build_prompt
from .providers import Provider
from .tools import ToolError, ToolRegistry, tool_error_json


class AgentError(Exception):
    """表示 Agent 流程执行失败。"""


class Agent:
    """协调历史消息、Provider 和四个基础工具的最小 Agent。"""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 3,
        max_history: int = 20,
    ) -> None:
        """创建 Agent 并配置模型、工具、循环和历史长度。"""

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
        """处理用户问题，必要时循环执行模型请求的工具调用。"""

        prompt = build_prompt(
            self.system_prompt,
            self.history,
            user_input,
            self.max_history,
        )
        input_items = [message.to_dict() for message in prompt]

        for _ in range(self.max_steps):
            try:
                response = self.provider.complete(
                    input_items,
                    self.tools.definitions(),
                )
            except Exception as exc:
                raise AgentError(f"model provider failed: {exc}") from exc

            if not response.tool_calls:
                self.history.extend(
                    [
                        Message(Role.USER, user_input),
                        Message(Role.ASSISTANT, response.text),
                    ]
                )
                return response.text

            input_items.extend(response.output_items)
            for tool_call in response.tool_calls:
                try:
                    output = self.tools.call(tool_call.name, tool_call.arguments)
                except ToolError as exc:
                    output = tool_error_json(str(exc))

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": output,
                    }
                )

        raise AgentError("maximum tool steps exceeded")
