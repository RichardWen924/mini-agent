import json
from pathlib import Path
import subprocess
from typing import Callable


class ToolError(Exception):
    """表示工具参数或工具执行失败。"""


ToolArguments = dict[str, object]
ToolDefinition = dict[str, object]
ToolFunction = Callable[[ToolArguments], str]


class ToolRegistry:
    """注册并执行四个基础命令行工具。"""

    def __init__(self, cwd: str | Path | None = None) -> None:
        """创建工具注册器，并固定工具的工作目录。"""

        self.cwd = Path(cwd or Path.cwd()).resolve()
        self._functions: dict[str, ToolFunction] = {
            "read": self._read,
            "bash": self._bash,
            "edit": self._edit,
            "write": self._write,
        }

    def definitions(self) -> list[ToolDefinition]:
        """返回可直接传给 Responses API 的工具定义。"""

        definitions = [
            {
                "type": "function",
                "name": "read",
                "description": "Read a text file in the working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "write",
                "description": "Create or overwrite a text file in the working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "edit",
                "description": "Replace one exact, unique string in a text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "bash",
                "description": "Run a shell command in the working directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        ]
        order = ("read", "bash", "edit", "write")
        definitions.sort(key=lambda definition: order.index(definition["name"]))
        return definitions

    def call(self, name: str, arguments: ToolArguments) -> str:
        """按名称执行工具，并返回给模型的文本结果。"""

        function = self._functions.get(name)
        if function is None:
            raise ToolError(f"unknown tool: {name}")

        return function(arguments)

    def _resolve_path(self, path: object) -> Path:
        """解析路径，并禁止访问工作目录之外的文件。"""

        if not isinstance(path, str) or not path:
            raise ToolError("path must be a non-empty string")

        resolved = (self.cwd / path).resolve()
        try:
            resolved.relative_to(self.cwd)
        except ValueError as exc:
            raise ToolError("path must stay inside the working directory") from exc

        return resolved

    @staticmethod
    def _string_argument(arguments: ToolArguments, name: str) -> str:
        """读取一个必需的字符串参数。"""

        value = arguments.get(name)
        if not isinstance(value, str):
            raise ToolError(f"{name} must be a string")
        return value

    def _read(self, arguments: ToolArguments) -> str:
        """读取文本文件，可选返回指定的闭区间行。"""

        path = self._resolve_path(arguments.get("path"))
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"read failed: {exc}") from exc

        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        if start_line is None and end_line is None:
            return content

        if not isinstance(start_line, int) or not isinstance(end_line, int):
            raise ToolError("start_line and end_line must be integers")
        if start_line < 1 or end_line < start_line:
            raise ToolError("line range must be a positive ascending range")

        lines = content.splitlines(keepends=True)
        return "".join(lines[start_line - 1 : end_line])

    def _write(self, arguments: ToolArguments) -> str:
        """创建或覆盖一个文本文件。"""

        path = self._resolve_path(arguments.get("path"))
        content = self._string_argument(arguments, "content")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"write failed: {exc}") from exc

        return f"wrote {path.relative_to(self.cwd)}"

    def _edit(self, arguments: ToolArguments) -> str:
        """在文件中执行一次精确且唯一的字符串替换。"""

        path = self._resolve_path(arguments.get("path"))
        old_string = self._string_argument(arguments, "old_string")
        new_string = self._string_argument(arguments, "new_string")
        if not old_string:
            raise ToolError("old_string must not be empty")

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"edit read failed: {exc}") from exc

        matches = content.count(old_string)
        if matches == 0:
            raise ToolError("old_string was not found")
        if matches > 1:
            raise ToolError("old_string must match exactly once")

        try:
            path.write_text(content.replace(old_string, new_string), encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"edit write failed: {exc}") from exc

        return f"edited {path.relative_to(self.cwd)}"

    def _bash(self, arguments: ToolArguments) -> str:
        """在固定工作目录执行 shell 命令，并限制最长运行时间。"""

        command = self._string_argument(arguments, "command")
        if not command.strip():
            raise ToolError("command must not be empty")

        try:
            result = subprocess.run(
                command,
                cwd=self.cwd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError("command timed out after 30 seconds") from exc
        except OSError as exc:
            raise ToolError(f"bash failed: {exc}") from exc

        output = f"{result.stdout}{result.stderr}"
        if result.returncode != 0:
            raise ToolError(f"command exited with code {result.returncode}: {output}")

        return output


def tool_error_json(error: str) -> str:
    """把工具错误转成模型可以读取的 JSON 文本。"""

    return json.dumps({"ok": False, "error": error}, ensure_ascii=False)
