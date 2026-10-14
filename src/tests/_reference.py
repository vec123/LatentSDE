"""
Access to the reference implementations in reference/ for parity tests.

A missing reference raises unittest.SkipTest, which pytest and run_all.py both report as a skip.
ROM-VISDE modules are imported without running visde/__init__.py, which needs pytorch_lightning and torchsde.
"""
import importlib
import sys
import types
import unittest
from pathlib import Path

import torch

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


def use_float64():
    torch.set_default_dtype(torch.float64)


def svise_module(name):
    """svise.<name> from reference/svise."""
    root = REFERENCE / "svise"
    if not (root / "svise").is_dir():
        raise unittest.SkipTest("reference/svise is missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module(f"svise.{name}")
    except ImportError as e:
        raise unittest.SkipTest(f"svise.{name} cannot be imported: {e}")


def visde_module(name):
    """visde.<name> from reference/rom-visde."""
    root = REFERENCE / "rom-visde" / "visde"
    if not root.is_dir():
        raise unittest.SkipTest("reference/rom-visde is missing")
    if "visde" not in sys.modules:
        package = types.ModuleType("visde")
        package.__path__ = [str(root)]
        sys.modules["visde"] = package
    try:
        return importlib.import_module(f"visde.{name}")
    except ImportError as e:
        raise unittest.SkipTest(f"visde.{name} cannot be imported: {e}")


def run_tests(namespace):
    """Runs every test_* function in namespace (a module's globals()); used by the __main__ block of each file."""
    failed = 0
    for name, test in list(namespace.items()):
        if not (name.startswith("test_") and callable(test)):
            continue
        try:
            test()
            print(f"  ok      {name}")
        except unittest.SkipTest as e:
            print(f"  skipped {name}: {e}")
        except Exception as e:  # report and continue
            failed += 1
            print(f"  FAILED  {name}: {type(e).__name__}: {e}")
    return failed
