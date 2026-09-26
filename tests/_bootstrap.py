from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_scripts_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
