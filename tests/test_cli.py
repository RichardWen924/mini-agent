from argparse import Namespace
from io import StringIO
import subprocess
import sys
import unittest
from unittest.mock import patch

from mini_agent.agent import AgentError
from mini_agent.cli import (
    create_agent,
    main,
    parse_args,
    run_interactive,
)
from mini_agent.providers import RuleBasedProvider
from mini_agent.tools import ToolRegistry


class FakeAgent:
    """记录输入并返回固定结果的测试 Agent。"""

    def __init__(self, response: str = "测试回复") -> None:
        """初始化测试回复和输入记录。"""

        self.response = response
        self.inputs: list[str] = []

    def ask(self, user_input: str) -> str:
        """记录用户输入并返回固定回复。"""

        self.inputs.append(user_input)
        return self.response


class ErrorAgent:
    """总是抛出 AgentError 的测试 Agent。"""

    def ask(self, user_input: str) -> str:
        """模拟 Agent 处理失败。"""

        raise AgentError("test failure")


class CliTests(unittest.TestCase):
    def test_cli_module_starts_successfully(self) -> None:
        """验证 CLI 模块可以启动并输出就绪提示。"""

        result = subprocess.run(
            [sys.executable, "-m", "mini_agent.cli"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mini-agent ready", result.stdout)

    def test_parse_args_uses_defaults(self) -> None:
        """验证命令行参数使用默认配置。"""

        args = parse_args([])

        self.assertEqual(args.max_steps, 3)
        self.assertEqual(args.max_history, 20)

    def test_parse_args_accepts_custom_values(self) -> None:
        """验证命令行参数可以覆盖默认配置。"""

        args = parse_args(
            [
                "--max-steps",
                "1",
                "--max-history",
                "5",
                "--system-prompt",
                "测试系统提示词",
            ]
        )

        self.assertEqual(args.max_steps, 1)
        self.assertEqual(args.max_history, 5)
        self.assertEqual(args.system_prompt, "测试系统提示词")

    def test_create_agent_uses_rule_based_provider(self) -> None:
        """验证 CLI 创建的是本地规则 Provider Agent。"""

        args = Namespace(
            max_steps=2,
            max_history=5,
            system_prompt="系统提示词",
        )

        agent = create_agent(args)

        self.assertIsInstance(agent.provider, RuleBasedProvider)
        self.assertIsInstance(agent.tools, ToolRegistry)
        self.assertEqual(agent.max_steps, 2)
        self.assertEqual(agent.max_history, 5)

    def test_interactive_loop_returns_normal_answer(self) -> None:
        """验证交互循环可以处理普通输入并输出回复。"""

        agent = FakeAgent("收到：hello")
        output = StringIO()

        status = run_interactive(
            agent,
            StringIO("hello\nquit\n"),
            output,
            StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(agent.inputs, ["hello"])
        self.assertIn("收到：hello", output.getvalue())

    def test_interactive_loop_executes_calculation(self) -> None:
        """验证交互循环可以通过 Agent 完成计算工具调用。"""

        agent = create_agent(
            Namespace(
                max_steps=3,
                max_history=20,
                system_prompt="系统提示词",
            )
        )
        output = StringIO()

        status = run_interactive(
            agent,
            StringIO("calc add 2 3\nquit\n"),
            output,
            StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertIn("5.0", output.getvalue())

    def test_empty_input_is_not_sent_to_agent(self) -> None:
        """验证空输入不会调用 Agent。"""

        agent = FakeAgent()
        output = StringIO()

        run_interactive(
            agent,
            StringIO("   \nquit\n"),
            output,
            StringIO(),
        )

        self.assertEqual(agent.inputs, [])
        self.assertIn("请输入内容", output.getvalue())

    def test_quit_stops_loop(self) -> None:
        """验证 quit 命令会正常结束交互循环。"""

        agent = FakeAgent()

        status = run_interactive(
            agent,
            StringIO("quit\n后续输入\n"),
            StringIO(),
            StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(agent.inputs, [])

    def test_exit_stops_loop(self) -> None:
        """验证 exit 命令会正常结束交互循环。"""

        agent = FakeAgent()

        status = run_interactive(
            agent,
            StringIO("exit\n后续输入\n"),
            StringIO(),
            StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(agent.inputs, [])

    def test_eof_stops_loop(self) -> None:
        """验证输入流结束时交互循环会正常退出。"""

        agent = FakeAgent()

        status = run_interactive(
            agent,
            StringIO("hello\n"),
            StringIO(),
            StringIO(),
        )

        self.assertEqual(status, 0)
        self.assertEqual(agent.inputs, ["hello"])

    def test_agent_error_is_written_to_stderr(self) -> None:
        """验证 AgentError 会输出到错误流而不是导致崩溃。"""

        errors = StringIO()

        status = run_interactive(
            ErrorAgent(),
            StringIO("hello\nquit\n"),
            StringIO(),
            errors,
        )

        self.assertEqual(status, 0)
        self.assertIn("test failure", errors.getvalue())

    def test_main_creates_agent_and_runs_interactive_loop(self) -> None:
        """验证 main 会组合 Agent 并启动交互循环。"""

        fake_agent = FakeAgent()

        with (
            patch("mini_agent.cli.create_agent", return_value=fake_agent) as create,
            patch("mini_agent.cli.run_interactive", return_value=7) as run,
        ):
            status = main(["--max-steps", "1"])

        self.assertEqual(status, 7)
        create.assert_called_once()
        run.assert_called_once()
        self.assertIs(run.call_args.args[0], fake_agent)


if __name__ == "__main__":
    unittest.main()
