"""Rewrite `constraints.txt` from the current environment (`make constraints`, after upgrading packages).

The API image and CI install with `pip install -c constraints.txt ...`, so they run exactly the library
versions the model was trained, calibrated and tested with.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "constraints.txt"
HEADER = """\
# Exact versions the v1.0.0 model was trained, calibrated and tested with (pip freeze of the owner's
# .venv). The API image and CI install with `pip install -c constraints.txt ...`, so they run the
# same libraries. After upgrading packages locally, regenerate it with `make constraints`.
"""


def main() -> None:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    lines: list[str] = []
    for line in freeze:
        lines.append(line)
        if line.startswith("xgboost=="):
            # Linux installs the CPU-only build of the same release (see pyproject.toml)
            lines.append(line.replace("xgboost==", "xgboost-cpu==", 1))
    PATH.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {PATH.name}: {len(lines)} pins")


if __name__ == "__main__":
    main()
