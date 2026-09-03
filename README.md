# Mini Agent Demo

一个使用 Python 和 uv 实现的最小 Agent Demo。

项目不使用 LangChain 或 LangGraph，直接实现消息管理、Prompt 构造、Provider、工具调用、Agent 循环和 CLI 交互，用于学习 Python 基础和 Agent 的基本工作方式。

## Features

- Python 标准库实现
- uv 项目管理
- 本地确定性 Rule-Based Provider
- 安全四则运算工具
- 有限 Agent 循环，避免无限调用
- 标准库 `unittest` 测试

## Requirements

- Python 3.13+
- uv

## Quick Start

在项目根目录运行：

```bash
uv run python -m mini_agent.cli
```

启动后输入：

```text
hello
calc add 2 3
calc multiply 4 5
你好
quit
```

示例输出：

```text
mini-agent ready
Hello!
工具结果：{"ok": true, "result": 5.0}
工具结果：{"ok": true, "result": 20.0}
你说的是：你好
再见！
```

退出命令：

```text
quit
exit
```

查看命令行参数：

```bash
uv run python -m mini_agent.cli --help
```

## Project Structure

```text
src/mini_agent/
├── models.py       # 消息数据结构
├── prompts.py      # Prompt 和历史裁剪
├── tools.py        # 工具定义与执行
├── providers.py    # Provider 接口和本地 Provider
├── agent.py        # Agent 主循环
└── cli.py          # 命令行入口
```

## Tests

运行全部测试：

```bash
uv run python -m unittest discover -s tests -v
```

当前共有 48 个测试。

## Main Program Size

主程序共 **567 行**，按 `src/mini_agent/*.py` 的物理行数统计，包含空行和 docstring。

该统计不包含：

- `tests/` 测试代码
- Markdown 文档
- `pyproject.toml`
- `uv.lock`

## License

本项目用于学习和演示。
