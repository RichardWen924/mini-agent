import unittest

from mini_agent.models import Message, Role
from mini_agent.providers import (
    ProviderResponse,
    RuleBasedProvider,
    ToolCall,
)


class RuleBasedProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        """为每个测试创建一个独立的规则 Provider。"""

        self.provider = RuleBasedProvider()

    def test_rule_provider_returns_deterministic_hello(self) -> None:
        """验证 hello 输入会返回确定性的普通助手回复。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "hello"),
            ]
        )

        self.assertEqual(response.message.role, Role.ASSISTANT)
        self.assertEqual(response.message.content, "Hello!")
        self.assertIsNone(response.tool_call)

    def test_rule_provider_returns_tool_call_for_calculation(
        self,
    ) -> None:
        """验证计算命令会生成 calculator 工具调用。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "calc add 2 3"),
            ]
        )

        self.assertIsNotNone(response.tool_call)
        self.assertEqual(response.tool_call.name, "calculator")

    def test_tool_call_contains_expected_arguments(self) -> None:
        """验证工具调用包含正确的运算名称和参数。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "calc multiply 4 5"),
            ]
        )

        self.assertEqual(
            response.tool_call.arguments,
            {
                "operation": "multiply",
                "a": 4.0,
                "b": 5.0,
            },
        )

    def test_rule_provider_handles_unknown_text(self) -> None:
        """验证普通文本会返回确定性的回显回复。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "你好"),
            ]
        )

        self.assertEqual(response.message.content, "你说的是：你好")
        self.assertIsNone(response.tool_call)

    def test_invalid_calculation_command_does_not_crash(
        self,
    ) -> None:
        """验证无效计算命令不会导致 Provider 崩溃。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "calc add x 2"),
            ]
        )

        self.assertIsInstance(response, ProviderResponse)
        self.assertIsNone(response.tool_call)

    def test_empty_messages_return_assistant_response(self) -> None:
        """验证空消息列表也会返回合法助手响应。"""

        response = self.provider.complete([])

        self.assertEqual(response.message.role, Role.ASSISTANT)
        self.assertIsNone(response.tool_call)

    def test_tool_message_returns_assistant_response(self) -> None:
        """验证工具消息会被转换成助手可读的回复。"""

        response = self.provider.complete(
            [
                Message(Role.TOOL, '{"ok": true, "result": 5.0}'),
            ]
        )

        self.assertEqual(response.message.role, Role.ASSISTANT)
        self.assertIn("5.0", response.message.content)
        self.assertIsNone(response.tool_call)

    def test_tool_call_has_expected_type(self) -> None:
        """验证计算命令返回的工具调用对象类型正确。"""

        response = self.provider.complete(
            [
                Message(Role.USER, "calc subtract 8 3"),
            ]
        )

        self.assertIsInstance(response.tool_call, ToolCall)


if __name__ == "__main__":
    unittest.main()
