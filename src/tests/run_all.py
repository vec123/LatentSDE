"""
Runs every test module in this folder without pytest. From the repository root:

    python -m src.tests.run_all
"""
import importlib
import sys
from pathlib import Path

from ._reference import run_tests


def main():
    failed = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        print(path.stem)
        module = importlib.import_module(f"{__package__}.{path.stem}")
        failed += run_tests(vars(module))
    print(f"{failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
