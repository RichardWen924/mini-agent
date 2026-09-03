import tempfile
import unittest
from pathlib import Path

from mini_agent.tools import ToolError, ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        """为每个测试创建隔离的临时工作目录。"""

        self.temp_dir = tempfile.TemporaryDirectory()
        self.registry = ToolRegistry(self.temp_dir.name)

    def tearDown(self) -> None:
        """删除测试使用的临时工作目录。"""

        self.temp_dir.cleanup()

    def test_definitions_contain_pi_four_tools(self) -> None:
        """验证注册器只暴露 read、write、edit、bash 四个工具。"""

        names = [definition["name"] for definition in self.registry.definitions()]

        self.assertEqual(names, ["read", "bash", "edit", "write"])

    def test_write_and_read_file(self) -> None:
        """验证 write 创建文件，read 返回文件内容。"""

        self.registry.call("write", {"path": "notes.txt", "content": "第一行\n第二行"})

        result = self.registry.call("read", {"path": "notes.txt"})

        self.assertEqual(result, "第一行\n第二行")

    def test_read_supports_one_based_line_range(self) -> None:
        """验证 read 支持一开始的闭区间行号。"""

        Path(self.temp_dir.name, "notes.txt").write_text(
            "第一行\n第二行\n第三行\n",
            encoding="utf-8",
        )

        result = self.registry.call(
            "read",
            {"path": "notes.txt", "start_line": 2, "end_line": 2},
        )

        self.assertEqual(result, "第二行\n")

    def test_edit_requires_unique_old_string(self) -> None:
        """验证 edit 只允许唯一匹配的精确替换。"""

        Path(self.temp_dir.name, "notes.txt").write_text(
            "旧内容\n",
            encoding="utf-8",
        )

        result = self.registry.call(
            "edit",
            {"path": "notes.txt", "old_string": "旧内容", "new_string": "新内容"},
        )

        self.assertIn("edited", result)
        self.assertEqual(
            Path(self.temp_dir.name, "notes.txt").read_text(encoding="utf-8"),
            "新内容\n",
        )

    def test_edit_rejects_non_unique_old_string(self) -> None:
        """验证 edit 遇到多处匹配时会失败。"""

        Path(self.temp_dir.name, "notes.txt").write_text(
            "重复\n重复\n",
            encoding="utf-8",
        )

        with self.assertRaises(ToolError):
            self.registry.call(
                "edit",
                {"path": "notes.txt", "old_string": "重复", "new_string": "新"},
            )

    def test_paths_must_stay_inside_working_directory(self) -> None:
        """验证文件工具不能访问工作目录外的路径。"""

        with self.assertRaises(ToolError):
            self.registry.call("read", {"path": "../outside.txt"})

    def test_bash_runs_in_working_directory(self) -> None:
        """验证 bash 在注册器工作目录中执行命令。"""

        result = self.registry.call("bash", {"command": "pwd"})

        self.assertEqual(Path(result.strip()).resolve(), Path(self.temp_dir.name).resolve())

    def test_unknown_tool_is_rejected(self) -> None:
        """验证未知工具名称会被拒绝。"""

        with self.assertRaises(ToolError):
            self.registry.call("calculator", {})


if __name__ == "__main__":
    unittest.main()
