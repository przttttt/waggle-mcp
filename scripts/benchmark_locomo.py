from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waggle.locomo_benchmark import main

if __name__ == "__main__":
    sys.exit(main())
