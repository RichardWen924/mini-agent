# Mini Agent

一个使用 Python、uv 和 OpenAI 兼容 Responses API 实现的最小 Agent Demo。

项目不使用 LangChain 或 LangGraph，提供 Pi 风格的四个基础工具：`read`、`write`、`edit`、`bash`。

## Requirements

- Python 3.13+
- uv
- OpenAI 或 DeepSeek API Key

## Quick Start

```bash
uv sync
export OPENAI_API_KEY="你的 API Key"
export OPENAI_MODEL="你的模型名"
uv run python -m mini_agent.cli
```

使用 DeepSeek 时：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
uv run python -m mini_agent.cli
```

程序会自动使用 `https://api.deepseek.com` 和 `deepseek-v4-flash`。

启动后直接输入问题，例如：

```text
mini-agent ready
请读取 README.md 并总结项目
```

输入 `quit` 或 `exit` 退出。

## Project Structure

```text
src/mini_agent/
├── models.py       # Message 和 Role 数据结构
├── prompts.py      # 系统提示词、历史消息和当前问题的拼接
├── tools.py        # read、write、edit、bash 四个工具
├── providers.py    # OpenAI Responses API 适配器
├── agent.py        # 上下文管理和工具调用循环
└── cli.py          # 命令行入口
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

测试使用 fake client 和临时目录，不会消耗 API 配额。

## Main Program Size

主程序共 **643 行**，按 `src/mini_agent/*.py` 的物理行数统计，包含空行和 docstring，不包含测试、Markdown 文档、`pyproject.toml` 和 `uv.lock`。

## License

本项目用于学习和演示。
