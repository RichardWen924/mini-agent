import unittest

from mini_agent.agent import Agent, AgentError
from mini_agent.models import Message, Role
from mini_agent.providers import ProviderResponse, ToolCall
from mini_agent.tools import ToolRegistry


class FakeProvider:
    """按顺序返回预设响应，并记录每次收到的消息。"""

    def __init__(self, responses: list[ProviderResponse]) -> None:
        """创建一个带有预设响应的 fake provider。"""

        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> ProviderResponse:
        """记录消息并返回下一条预设响应。"""

        self.calls.append(list(messages))

        if not self.responses:
            raise AssertionError("fake provider has no response left")

        return self.responses.pop(0)


class FailingProvider:
    """总是抛出异常的 fake provider。"""

    def complete(self, messages: list[Message]) -> ProviderResponse:
        """模拟模型服务调用失败。"""

        raise RuntimeError("provider failure")


def assistant_response(content: str) -> ProviderResponse:
    """创建一条普通助手响应，减少测试中的重复代码。"""

    return ProviderResponse(Message(Role.ASSISTANT, content))


def calculator_response(
    operation: str,
    a: float,
    b: float,
) -> ProviderResponse:
    """创建一条 calculator 工具调用响应。"""

    return ProviderResponse(
        message=Message(Role.ASSISTANT, "我将调用计算器。"),
        tool_call=ToolCall(
            name="calculator",
            arguments={
                "operation": operation,
                "a": a,
                "b": b,
            },
        ),
    )


class AgentTests(unittest.TestCase):
    def create_agent(
        self,
        provider: object,
        max_steps: int = 3,
    ) -> Agent:
        """使用测试默认配置创建 Agent。"""

        return Agent(
            provider=provider,
            tools=ToolRegistry(),
            system_prompt="你是一个测试助手。",
            max_steps=max_steps,
            max_history=20,
        )

    def test_normal_response_calls_provider_once(self) -> None:
        """验证普通回复只调用 Provider 一次。"""

        provider = FakeProvider([assistant_response("你好！")])
        agent = self.create_agent(provider)

        result = agent.ask("hello")

        self.assertEqual(result, "你好！")
        self.assertEqual(len(provider.calls), 1)

    def test_normal_response_is_saved_to_history(self) -> None:
        """验证普通问答的用户消息和助手消息都会保存。"""

        provider = FakeProvider([assistant_response("回答")])
        agent = self.create_agent(provider)

        agent.ask("问题")

        self.assertEqual(
            agent.history,
            [
                Message(Role.USER, "问题"),
                Message(Role.ASSISTANT, "回答"),
            ],
        )

    def test_tool_call_executes_tool(self) -> None:
        """验证 Agent 会执行 Provider 请求的工具。"""

        provider = FakeProvider(
            [
                calculator_response("add", 2, 3),
                assistant_response("结果是 5。"),
            ]
        )
        agent = self.create_agent(provider)

        result = agent.ask("calc add 2 3")

        self.assertEqual(result, "结果是 5。")
        self.assertEqual(len(provider.calls), 2)

    def test_tool_result_is_added_to_history(self) -> None:
        """验证工具结果会以工具消息保存到历史。"""

        provider = FakeProvider(
            [
                calculator_response("add", 2, 3),
                assistant_response("结果是 5。"),
            ]
        )
        agent = self.create_agent(provider)

        agent.ask("calc add 2 3")

        self.assertEqual(agent.history[2].role, Role.TOOL)
        self.assertIn('"result": 5.0', agent.history[2].content)

    def test_tool_call_requests_second_provider_response(self) -> None:
        """验证工具执行后会把工具结果发送给 Provider。"""

        provider = FakeProvider(
            [
                calculator_response("add", 2, 3),
                assistant_response("结果是 5。"),
            ]
        )
        agent = self.create_agent(provider)

        agent.ask("calc add 2 3")

        second_call = provider.calls[1]
        self.assertEqual(second_call[-1].role, Role.TOOL)
        self.assertIn('"result": 5.0', second_call[-1].content)

    def test_tool_error_is_sent_back_to_provider(self) -> None:
        """验证工具失败时错误信息会发送回 Provider。"""

        provider = FakeProvider(
            [
                calculator_response("divide", 1, 0),
                assistant_response("除数不能为零。"),
            ]
        )
        agent = self.create_agent(provider)

        result = agent.ask("calc divide 1 0")

        self.assertEqual(result, "除数不能为零。")
        self.assertIn("cannot divide by zero", provider.calls[1][-1].content)

    def test_max_steps_prevents_infinite_loop(self) -> None:
        """验证最大步数可以阻止工具调用无限循环。"""

        provider = FakeProvider(
            [
                calculator_response("add", 1, 2),
                calculator_response("add", 3, 4),
            ]
        )
        agent = self.create_agent(provider, max_steps=2)

        result = agent.ask("calc add 1 2")

        self.assertIn("达到最大处理步骤", result)
        self.assertEqual(len(provider.calls), 2)

    def test_zero_max_steps_raises_value_error(self) -> None:
        """验证最大步数为零时不能创建 Agent。"""

        with self.assertRaises(ValueError):
            Agent(
                provider=FakeProvider([]),
                tools=ToolRegistry(),
                system_prompt="系统指令",
                max_steps=0,
            )

    def test_provider_error_becomes_agent_error(self) -> None:
        """验证 Provider 异常会被转换为 AgentError。"""

        agent = self.create_agent(FailingProvider())

        with self.assertRaises(AgentError):
            agent.ask("hello")

    def test_user_message_is_added_only_once(self) -> None:
        """验证同一轮用户输入不会在首次 prompt 中重复。"""

        provider = FakeProvider([assistant_response("回答")])
        agent = self.create_agent(provider)

        agent.ask("问题")

        user_messages = [
            message
            for message in provider.calls[0]
            if message.role == Role.USER and message.content == "问题"
        ]
        self.assertEqual(len(user_messages), 1)


if __name__ == "__main__":
    unittest.main()
