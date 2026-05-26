import subprocess
import sys
from pathlib import Path
import unittest


class QuickTests(unittest.TestCase):
    def test_main_exists(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "main.py").exists())

    def test_main_help(self):
        root = Path(__file__).resolve().parents[1]
        p = subprocess.run([sys.executable, "main.py", "--help"], cwd=str(root))
        self.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
