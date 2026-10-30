"""Write the API's OpenAPI schema to `frontend/openapi.json`, the contract the UI's types come from.

`make api-types` runs this, then generates `frontend/src/lib/api-types.ts` from the file. Both are
committed, so an API change shows up as a reviewable diff; CI regenerates them and fails on drift.
Offline: building the schema needs neither the database nor Redis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python scripts/export_openapi.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import create_app

OPENAPI_PATH = PROJECT_ROOT / "frontend" / "openapi.json"


def openapi_text() -> str:
    return json.dumps(create_app().openapi(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    OPENAPI_PATH.write_text(openapi_text(), encoding="utf-8")
    print(f"wrote {OPENAPI_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
