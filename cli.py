from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    pkg_root = Path(__file__).resolve().parent
    benchmark_dir = pkg_root / "benchmark"
    benchmark_main = benchmark_dir / "main.py"

    sys.path.insert(0, str(benchmark_dir))
    try:
        runpy.run_path(str(benchmark_main), run_name="__main__")
    finally:
        if sys.path and sys.path[0] == str(benchmark_dir):
            sys.path.pop(0)
