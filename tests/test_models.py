from dataclasses import FrozenInstanceError
import unittest

from mini_agent.models import Message, Role


class MessageTests(unittest.TestCase):
    def test_user_message_can_be_created(self) -> None:
        """验证用户消息可以保存角色和文本内容。"""

        message = Message(Role.USER, "你好")

        self.assertEqual(message.role, Role.USER)
        self.assertEqual(message.content, "你好")

    def test_message_to_dict_uses_role_value(self) -> None:
        """验证消息字典使用角色值而不是枚举名称。"""

        message = Message(Role.USER, "你好")

        self.assertEqual(
            message.to_dict(),
            {
                "role": "user",
                "content": "你好",
            },
        )

    def test_empty_content_is_detected(self) -> None:
        """验证空字符串会被识别为空消息。"""

        self.assertTrue(Message(Role.USER, "").is_empty())

    def test_whitespace_content_is_detected(self) -> None:
        """验证只包含空白字符的内容会被识别为空消息。"""

        self.assertTrue(Message(Role.USER, "   \t\n").is_empty())

    def test_normal_content_is_not_empty(self) -> None:
        """验证普通文本不会被识别为空消息。"""

        self.assertFalse(Message(Role.ASSISTANT, "你好").is_empty())

    def test_invalid_role_raises_type_error(self) -> None:
        """验证传入非 Role 值时会抛出类型错误。"""

        with self.assertRaises(TypeError):
            Message("user", "你好")  # type: ignore[arg-type]

    def test_invalid_content_raises_type_error(self) -> None:
        """验证传入非字符串内容时会抛出类型错误。"""

        with self.assertRaises(TypeError):
            Message(Role.USER, 123)  # type: ignore[arg-type]

    def test_message_is_immutable(self) -> None:
        """验证消息创建后不能修改字段。"""

        message = Message(Role.USER, "你好")

        with self.assertRaises(FrozenInstanceError):
            message.content = "新的内容"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
