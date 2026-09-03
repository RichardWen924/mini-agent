import json
import tempfile
import unittest
from pathlib import Path

from mini_agent.agent import Agent, AgentError
from mini_agent.models import Message, Role
from mini_agent.providers import ProviderResponse, ToolCall
from mini_agent.tools import ToolRegistry


class FakeProvider:
    """按顺序返回预设响应，并记录收到的请求。"""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        """创建一个带有预设响应的 fake provider。"""

        self.responses = list(responses)
        self.calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    def complete(
        self,
        input_items: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> ProviderResponse:
        """记录请求和工具定义，并返回下一条预设响应。"""

        self.calls.append((list(input_items), list(tools)))

        if not self.responses:
            raise AssertionError("fake provider has no response left")

        return self.responses.pop(0)


class FailingProvider:
    """总是抛出异常的 fake provider。"""

    def complete(
        self,
        input_items: list[dict[str, object]],
        tools: list[dict[str, object]],
    ) -> ProviderResponse:
        """模拟模型服务调用失败。"""

        raise RuntimeError("provider failure")


def text_response(content: str) -> ProviderResponse:
    """创建一条普通文本响应。"""

    return ProviderResponse(text=content)


class AgentTests(unittest.TestCase):
    def create_agent(
        self,
        provider: object,
        max_steps: int = 3,
        max_history: int = 20,
        tools: ToolRegistry | None = None,
    ) -> Agent:
        """使用测试默认配置创建 Agent。"""

        return Agent(
            provider=provider,
            tools=tools or ToolRegistry(),
            system_prompt="你是一个测试助手。",
            max_steps=max_steps,
            max_history=max_history,
        )

    def test_ask_returns_provider_text(self) -> None:
        """验证 Agent 返回 Provider 生成的文本。"""

        provider = FakeProvider([text_response("你好！")])
        agent = self.create_agent(provider)

        self.assertEqual(agent.ask("hello"), "你好！")
        self.assertEqual(len(provider.calls), 1)

    def test_first_request_contains_prompt_and_all_tool_definitions(self) -> None:
        """验证首次请求包含 Prompt 和四个工具定义。"""

        provider = FakeProvider([text_response("回答")])
        agent = self.create_agent(provider)

        agent.ask("问题")

        input_items, tool_definitions = provider.calls[0]
        self.assertEqual(
            input_items,
            [
                {"role": "system", "content": "你是一个测试助手。"},
                {"role": "user", "content": "问题"},
            ],
        )
        self.assertEqual(
            [tool["name"] for tool in tool_definitions],
            ["read", "bash", "edit", "write"],
        )

    def test_conversation_history_is_sent_on_next_request(self) -> None:
        """验证下一轮请求会携带上一轮对话历史。"""

        provider = FakeProvider(
            [text_response("第一轮回答"), text_response("第二轮回答")]
        )
        agent = self.create_agent(provider)

        agent.ask("第一轮问题")
        agent.ask("第二轮问题")

        self.assertEqual(
            provider.calls[1][0],
            [
                {"role": "system", "content": "你是一个测试助手。"},
                {"role": "user", "content": "第一轮问题"},
                {"role": "assistant", "content": "第一轮回答"},
                {"role": "user", "content": "第二轮问题"},
            ],
        )

    def test_tool_call_is_executed_and_result_is_sent_back(self) -> None:
        """验证 Agent 会执行工具并提交 function_call_output。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            path.write_text("工具读取成功", encoding="utf-8")
            provider = FakeProvider(
                [
                    ProviderResponse(
                        text="",
                        tool_calls=(
                            ToolCall(
                                call_id="call_1",
                                name="read",
                                arguments={"path": "note.txt"},
                            ),
                        ),
                        output_items=(
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "read",
                                "arguments": '{"path":"note.txt"}',
                            },
                        ),
                    ),
                    text_response("文件内容已读取。"),
                ]
            )
            agent = self.create_agent(provider, tools=ToolRegistry(directory))

            result = agent.ask("读取 note.txt")

        self.assertEqual(result, "文件内容已读取。")
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            provider.calls[1][0][-1],
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "工具读取成功",
            },
        )

    def test_tool_error_is_sent_back_as_json(self) -> None:
        """验证工具失败时 Agent 会把错误信息交给模型。"""

        provider = FakeProvider(
            [
                ProviderResponse(
                    text="",
                    tool_calls=(
                        ToolCall(
                            call_id="call_1",
                            name="missing",
                            arguments={},
                        ),
                    ),
                    output_items=(),
                ),
                text_response("工具不存在。"),
            ]
        )
        agent = self.create_agent(provider)

        self.assertEqual(agent.ask("调用不存在的工具"), "工具不存在。")
        error_output = json.loads(provider.calls[1][0][-1]["output"])
        self.assertFalse(error_output["ok"])
        self.assertIn("unknown tool", error_output["error"])

    def test_max_steps_prevents_infinite_tool_loop(self) -> None:
        """验证最大步骤可以阻止工具调用无限循环。"""

        tool_call = ToolCall("call_1", "read", {"path": "missing.txt"})
        provider = FakeProvider(
            [
                ProviderResponse(text="", tool_calls=(tool_call,)),
                ProviderResponse(text="", tool_calls=(tool_call,)),
            ]
        )
        agent = self.create_agent(provider, max_steps=2)

        with self.assertRaises(AgentError):
            agent.ask("持续调用工具")

        self.assertEqual(len(provider.calls), 2)

    def test_provider_error_becomes_agent_error(self) -> None:
        """验证 Provider 异常会被转换为 AgentError。"""

        agent = self.create_agent(FailingProvider())

        with self.assertRaisesRegex(AgentError, "provider failure"):
            agent.ask("hello")

        self.assertEqual(agent.history, [])

    def test_invalid_limits_and_system_prompt_raise_value_error(self) -> None:
        """验证系统提示词和循环限制必须有效。"""

        with self.assertRaises(ValueError):
            Agent(provider=FakeProvider([]), tools=ToolRegistry(), system_prompt="   ")

        with self.assertRaises(ValueError):
            Agent(
                provider=FakeProvider([]),
                tools=ToolRegistry(),
                system_prompt="系统指令",
                max_steps=0,
            )

        with self.assertRaises(ValueError):
            Agent(
                provider=FakeProvider([]),
                tools=ToolRegistry(),
                system_prompt="系统指令",
                max_history=-1,
            )


if __name__ == "__main__":
    unittest.main()
