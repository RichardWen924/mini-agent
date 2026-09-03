import argparse
from collections.abc import Sequence
import sys
from typing import TextIO

from .agent import Agent, AgentError
from .providers import OpenAIProvider
from .tools import ToolRegistry


DEFAULT_SYSTEM_PROMPT = "你是一个有帮助的助手。"
EXIT_COMMANDS = {"exit", "quit"}


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """解析命令行参数并返回程序配置。"""

    parser = argparse.ArgumentParser(
        description="运行一个使用 OpenAI API 和四个基础工具的最小 Agent。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="使用的模型；未指定时读取 OPENAI_MODEL。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="每个问题允许的最大工具调用轮数。",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=20,
        help="发送给模型的最大历史消息数量。",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Agent 使用的系统提示词。",
    )
    return parser.parse_args(argv)


def create_agent(args: argparse.Namespace) -> Agent:
    """根据命令行配置创建 OpenAI Agent。"""

    return Agent(
        provider=OpenAIProvider(model=args.model),
        tools=ToolRegistry(),
        system_prompt=args.system_prompt,
        max_steps=args.max_steps,
        max_history=args.max_history,
    )


def run_interactive(
    agent: Agent,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> int:
    """读取用户输入、调用 Agent 并输出回复。"""

    print("mini-agent ready", file=output_stream)

    while True:
        try:
            line = input_stream.readline()
        except KeyboardInterrupt:
            print("\n再见！", file=output_stream)
            return 0

        if line == "":
            print("再见！", file=output_stream)
            return 0

        user_input = line.strip()

        if not user_input:
            print("请输入内容。", file=output_stream)
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("再见！", file=output_stream)
            return 0

        try:
            answer = agent.ask(user_input)
        except AgentError as exc:
            print(f"错误：{exc}", file=error_stream)
            continue

        print(answer, file=output_stream)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """解析参数、创建 Agent 并启动交互循环。"""

    args = parse_args(argv)

    try:
        agent = create_agent(args)
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    return run_interactive(
        agent,
        sys.stdin,
        sys.stdout,
        sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
