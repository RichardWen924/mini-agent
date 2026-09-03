# Mini Agent

一个使用 Python 和 uv 构建的最小 Agent Demo。

这个项目不使用 LangChain、LangGraph 或其他 Agent 框架，直接使用 Python 标准库实现 Agent 的基本组成：消息结构、Prompt 构造、模型适配器、工具调用、有限循环和命令行交互。

项目的主要目的是学习 Python 基础，同时理解一个最小 Agent 从输入到输出的完整数据链路。

## 项目特点

- Python 3.13+
- 使用 uv 管理项目和运行命令
- 不依赖 LangChain 或 LangGraph
- 使用 `dataclass` 表示消息和响应
- 使用 `Protocol` 抽象模型 Provider
- 内置确定性的本地 Rule-Based Provider
- 内置安全的四则运算工具
- 工具调用有参数校验，不使用 `eval()` 或 `exec()`
- Agent 具有最大处理步数，避免无限循环
- 使用标准库 `unittest` 编写测试
- 支持命令行交互

## 快速开始

确认已经安装 Python 和 uv，然后在项目根目录执行：

```bash
uv run python -m mini_agent.cli
```

启动后可以输入：

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

## 命令行参数

查看帮助：

```bash
uv run python -m mini_agent.cli --help
```

可用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--max-steps` | `3` | 每个问题允许的最大模型处理步数 |
| `--max-history` | `20` | 发送给模型的最大历史消息数量 |
| `--system-prompt` | 内置中文提示词 | Agent 使用的系统提示词 |

例如：

```bash
uv run python -m mini_agent.cli --max-steps 1 --max-history 5
```

也可以使用管道输入：

```bash
printf 'hello\ncalc add 2 3\nquit\n' | uv run python -m mini_agent.cli
```

## 项目结构

```text
Mini-agent/
├── pyproject.toml
├── uv.lock
├── README.md
├── src/
│   └── mini_agent/
│       ├── __init__.py
│       ├── agent.py
│       ├── cli.py
│       ├── models.py
│       ├── prompts.py
│       ├── providers.py
│       ├── py.typed
│       └── tools.py
├── tests/
│   ├── test_agent.py
│   ├── test_cli.py
│   ├── test_models.py
│   ├── test_prompts.py
│   └── test_providers.py
└── docs/
    └── mini-agent-architecture/
        └── ARCHITECTURE.md
```

## 文件职责

| 文件 | 作用 |
|---|---|
| `src/mini_agent/__init__.py` | Python 包初始化文件；当前保留 uv 初始化时生成的示例函数，不参与 Agent 主链路 |
| `src/mini_agent/models.py` | 定义 `Role` 和不可变的 `Message` 数据结构 |
| `src/mini_agent/prompts.py` | 组合系统提示词、历史消息和当前问题，并裁剪历史长度 |
| `src/mini_agent/tools.py` | 实现安全计算器、工具异常和工具注册器 |
| `src/mini_agent/providers.py` | 定义 Provider 接口、工具调用响应和本地规则 Provider |
| `src/mini_agent/agent.py` | 协调 Prompt、Provider 和 ToolRegistry，执行有限 Agent 循环 |
| `src/mini_agent/cli.py` | 解析命令行参数并运行交互式输入循环 |
| `src/mini_agent/py.typed` | 标记该包包含可供类型检查器使用的类型信息 |

测试文件与源文件一一对应：

| 测试文件 | 覆盖内容 |
|---|---|
| `tests/test_models.py` | 消息角色、类型校验、不可变性和字典转换 |
| `tests/test_prompts.py` | Prompt 顺序、历史裁剪、输入校验和副作用 |
| `tests/test_tools.py` | 四则运算、安全校验、工具注册和 JSON 结果 |
| `tests/test_providers.py` | 规则 Provider、普通回复和工具调用响应 |
| `tests/test_agent.py` | Agent 状态流、工具循环、异常和步数限制 |
| `tests/test_cli.py` | 参数解析、交互输入、退出和错误输出 |

## 数据链路

### 普通回复

```text
用户输入
  ↓
cli.run_interactive()
  ↓
Agent.ask()
  ↓
prompts.build_prompt()
  ↓
Provider.complete()
  ↓
ProviderResponse(message=assistant, tool_call=None)
  ↓
保存助手消息到 history
  ↓
CLI 输出最终文本
```

### 工具调用

```text
用户输入：calc add 2 3
  ↓
RuleBasedProvider 解析命令
  ↓
返回 ToolCall(name="calculator", arguments={...})
  ↓
Agent 识别 tool_call
  ↓
ToolRegistry.call()
  ↓
calculate() 校验参数并执行
  ↓
生成 Role.TOOL 消息
  ↓
再次调用 Provider
  ↓
返回最终助手消息
  ↓
CLI 输出结果
```

一次完整的工具对话历史大致如下：

```python
[
    Message(Role.USER, "calc add 2 3"),
    Message(Role.ASSISTANT, "我将调用计算器。"),
    Message(Role.TOOL, '{"ok": true, "result": 5.0}'),
    Message(Role.ASSISTANT, "工具结果：..."),
]
```

## 安全边界

计算器只允许预先定义的四种操作：

```text
add
subtract
multiply
divide
```

工具层会拒绝：

- 未知操作
- 缺少参数
- 多余参数
- 字符串操作数
- 布尔操作数
- 无穷大或非有限数字
- 除数为零

项目不会把用户输入直接当作 Python 代码执行。

## 测试

运行全部测试：

```bash
uv run python -m unittest discover -s tests -v
```

编译检查：

```bash
uv run python -m compileall -q src tests
```

检查 diff 空白错误：

```bash
git diff --check
```

当前项目包含 48 个测试，覆盖从数据结构到命令行入口的主要行为。

## 主程序行数

主程序行数按 `src/mini_agent/*.py` 的物理行数统计，包含空行和 docstring，不包含 `tests/`、Markdown 文档、配置文件和锁文件。

| 文件 | 行数 |
|---|---:|
| `__init__.py` | 2 |
| `models.py` | 36 |
| `prompts.py` | 41 |
| `tools.py` | 142 |
| `providers.py` | 140 |
| `agent.py` | 89 |
| `cli.py` | 117 |
| **主程序合计** | **567** |

## 当前范围

这是一个学习型最小 Demo，不是生产级 Agent 框架。当前默认使用本地确定性 Provider，因此不需要 API Key，也不会访问外部模型服务。

真实模型接入可以作为后续扩展，通过新增 Provider 实现完成，而不需要改变 Agent 主循环。

## 设计原则

- 保持每个模块职责单一。
- 通过接口隔离模型供应商。
- 将工具安全校验放在工具层。
- 将流程控制放在 Agent 层。
- 使用可重复的本地 Provider 测试核心逻辑。
- 优先使用标准库，避免不必要的框架依赖。
