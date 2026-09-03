from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mini_agent.models import Message, Role
from mini_agent.providers import OpenAIProvider


class FakeResponses:
    """记录 Responses API 调用的 fake responses 资源。"""

    def __init__(self, output_text: str, output: list[object] | None = None) -> None:
        """保存 fake 模型输出、工具调用和调用参数。"""

        self.output_text = output_text
        self.output = output or []
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        """记录请求参数并返回假响应。"""

        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text, output=self.output)


class FakeClient:
    """提供 fake Responses API 的客户端。"""

    def __init__(self, response: FakeResponses) -> None:
        """初始化 fake responses 资源。"""

        self.responses = response


class OpenAIProviderTests(unittest.TestCase):
    @patch("mini_agent.providers.OpenAI")
    def test_deepseek_key_selects_deepseek_endpoint_and_model(
        self,
        openai_class: object,
    ) -> None:
        """验证设置 DEEPSEEK_API_KEY 时使用 DeepSeek 配置。"""

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "ds-key"}, clear=True):
            provider = OpenAIProvider()

        openai_class.assert_called_once_with(  # type: ignore[attr-defined]
            api_key="ds-key",
            base_url="https://api.deepseek.com",
        )
        self.assertEqual(provider.model, "deepseek-v4-flash")

    @patch("mini_agent.providers.OpenAI")
    def test_base_url_can_be_configured_for_openai_compatible_api(
        self,
        openai_class: object,
    ) -> None:
        """验证可以通过 OPENAI_BASE_URL 配置兼容 API 地址。"""

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "compatible-key",
                "OPENAI_BASE_URL": "https://example.test/v1",
                "OPENAI_MODEL": "compatible-model",
            },
            clear=True,
        ):
            provider = OpenAIProvider()

        openai_class.assert_called_once_with(  # type: ignore[attr-defined]
            api_key="compatible-key",
            base_url="https://example.test/v1",
        )
        self.assertEqual(provider.model, "compatible-model")

    def test_complete_calls_responses_api_with_tools(self) -> None:
        """验证 Provider 使用 Responses API 并传递工具定义。"""

        response = FakeResponses("你好！")
        provider = OpenAIProvider(
            model="test-model",
            client=FakeClient(response),  # type: ignore[arg-type]
        )
        input_items = [{"role": "user", "content": "你好"}]
        tool_definitions = [{"type": "function", "name": "read"}]

        result = provider.complete(input_items, tool_definitions)

        self.assertEqual(result.text, "你好！")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(
            response.calls,
            [
                {
                    "model": "test-model",
                    "input": input_items,
                    "tools": tool_definitions,
                }
            ],
        )

    def test_complete_parses_function_call_output(self) -> None:
        """验证 Provider 会解析模型返回的 function_call。"""

        function_call = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="read",
            arguments='{"path":"README.md"}',
        )
        response = FakeResponses("", [function_call])
        provider = OpenAIProvider(
            model="test-model",
            client=FakeClient(response),  # type: ignore[arg-type]
        )

        result = provider.complete(
            [{"role": "user", "content": "读取 README"}],
            [],
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].call_id, "call_1")
        self.assertEqual(result.tool_calls[0].name, "read")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "README.md"})
        self.assertEqual(result.output_items[0]["type"], "function_call")

    def test_model_can_be_read_from_environment(self) -> None:
        """验证未显式传入模型时可以读取 OPENAI_MODEL。"""

        response = FakeResponses("回答")

        with patch.dict("os.environ", {"OPENAI_MODEL": "env-model"}):
            provider = OpenAIProvider(client=FakeClient(response))  # type: ignore[arg-type]
            provider.complete([{"role": "user", "content": "问题"}], [])

        self.assertEqual(response.calls[0]["model"], "env-model")

    def test_empty_input_is_rejected(self) -> None:
        """验证 Provider 不会发送空输入。"""

        provider = OpenAIProvider(client=FakeClient(FakeResponses("回答")))  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            provider.complete([], [])


if __name__ == "__main__":
    unittest.main()
