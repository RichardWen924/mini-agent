import unittest

from mini_agent.models import Message, Role
from mini_agent.prompts import build_prompt, trim_history


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        """为每个测试准备一组不会被测试修改的历史消息。"""

        self.history = [
            Message(Role.USER, "旧问题"),
            Message(Role.ASSISTANT, "旧回答"),
            Message(Role.USER, "更旧的问题"),
        ]

    def test_system_message_is_first(self) -> None:
        """验证系统消息始终位于 prompt 第一条。"""

        prompt = build_prompt("你是一个助手", self.history, "新问题", 10)

        self.assertEqual(prompt[0], Message(Role.SYSTEM, "你是一个助手"))

    def test_history_order_is_preserved(self) -> None:
        """验证保留的历史消息顺序与原始顺序一致。"""

        prompt = build_prompt("系统指令", self.history, "新问题", 10)

        self.assertEqual(prompt[1:4], self.history)

    def test_current_user_message_is_added_once(self) -> None:
        """验证当前用户问题只被添加一次。"""

        prompt = build_prompt("系统指令", self.history, "新问题", 10)

        self.assertEqual(prompt[-1], Message(Role.USER, "新问题"))
        self.assertEqual(
            sum(message.content == "新问题" for message in prompt),
            1,
        )

    def test_trim_history_keeps_latest_messages(self) -> None:
        """验证历史裁剪保留最近的消息。"""

        trimmed = trim_history(self.history, 2)

        self.assertEqual(trimmed, self.history[-2:])

    def test_zero_max_history_returns_no_history(self) -> None:
        """验证历史上限为零时只保留系统消息和当前问题。"""

        prompt = build_prompt("系统指令", self.history, "新问题", 0)

        self.assertEqual(
            prompt,
            [
                Message(Role.SYSTEM, "系统指令"),
                Message(Role.USER, "新问题"),
            ],
        )

    def test_negative_max_history_raises_value_error(self) -> None:
        """验证负数历史上限会抛出 ValueError。"""

        with self.assertRaises(ValueError):
            trim_history(self.history, -1)

    def test_build_prompt_does_not_mutate_history(self) -> None:
        """验证构造 prompt 不会修改传入的历史列表。"""

        original_history = list(self.history)

        build_prompt("系统指令", self.history, "新问题", 2)

        self.assertEqual(self.history, original_history)

    def test_empty_system_prompt_raises_value_error(self) -> None:
        """验证空系统提示词会抛出 ValueError。"""

        with self.assertRaises(ValueError):
            build_prompt("   ", self.history, "新问题", 10)

    def test_empty_user_input_raises_value_error(self) -> None:
        """验证空用户输入会抛出 ValueError。"""

        with self.assertRaises(ValueError):
            build_prompt("系统指令", self.history, "\t", 10)


if __name__ == "__main__":
    unittest.main()
