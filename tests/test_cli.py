import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_cli_module_starts_successfully(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mini_agent.cli"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mini-agent ready", result.stdout)


if __name__ == "__main__":
    unittest.main()
