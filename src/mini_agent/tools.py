from collections.abc import Callable, Mapping
import json
import math
from operator import add, mul, sub, truediv
from typing import TypeAlias


ToolHandler: TypeAlias = Callable[[Mapping[str, object]], float]


class ToolError(Exception):
    """表示工具调用过程中的输入或执行错误。"""


def _validate_calculator_arguments(
    arguments: Mapping[str, object],
) -> None:
    """检查计算器参数是否包含必需字段且没有额外字段。"""

    if not isinstance(arguments, Mapping):
        raise ToolError("arguments must be a mapping")

    required = {"operation", "a", "b"}
    provided = set(arguments)

    missing = required - provided
    if missing:
        raise ToolError(f"missing arguments: {sorted(missing)}")

    unexpected = provided - required
    if unexpected:
        raise ToolError(f"unexpected arguments: {sorted(unexpected)}")


def _get_number(
    arguments: Mapping[str, object],
    name: str,
) -> float:
    """读取并校验一个有限的数字参数。"""

    value = arguments[name]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be a number")

    number = float(value)

    if not math.isfinite(number):
        raise ToolError(f"{name} must be finite")

    return number


def calculate(arguments: Mapping[str, object]) -> float:
    """根据参数执行安全的四则运算。"""

    _validate_calculator_arguments(arguments)

    operation = arguments["operation"]

    if not isinstance(operation, str):
        raise ToolError("operation must be a string")

    a = _get_number(arguments, "a")
    b = _get_number(arguments, "b")

    operations: dict[str, Callable[[float, float], float]] = {
        "add": add,
        "subtract": sub,
        "multiply": mul,
        "divide": truediv,
    }

    if operation not in operations:
        raise ToolError(f"unknown operation: {operation}")

    if operation == "divide" and b == 0:
        raise ToolError("cannot divide by zero")

    result = operations[operation](a, b)

    if not math.isfinite(result):
        raise ToolError("calculation result must be finite")

    return result


class ToolRegistry:
    """保存工具处理函数，并根据工具名称分发调用。"""

    def __init__(self) -> None:
        """创建包含默认计算器工具的注册器。"""

        self._handlers: dict[str, ToolHandler] = {
            "calculator": calculate,
        }

    def call(
        self,
        name: str,
        arguments: Mapping[str, object],
    ) -> str:
        """调用指定工具，并将成功结果转换为 JSON 字符串。"""

        if name not in self._handlers:
            raise ToolError(f"unknown tool: {name}")

        result = self._handlers[name](arguments)

        return json.dumps(
            {
                "ok": True,
                "result": result,
            },
            ensure_ascii=False,
        )

    def schemas(self) -> list[dict[str, object]]:
        """返回工具名称、说明和参数结构供模型使用。"""

        return [
            {
                "name": "calculator",
                "description": "执行安全的四则运算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "add",
                                "subtract",
                                "multiply",
                                "divide",
                            ],
                        },
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["operation", "a", "b"],
                },
            }
        ]