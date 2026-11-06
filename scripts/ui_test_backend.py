"""Run the backend for UI testing with a throwaway data folder.

The real app data (data/local_ai.db, settings, sessions) is never touched: everything
goes to the folder given by --data (default: a temp folder). Models are read from
models/ as usual. Serves the built UI at http://127.0.0.1:<port>.

    python scripts/ui_test_backend.py --port 8765 --data C:\\path\\to\\scratch\\ui-data
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    data = Path(args.data) if args.data else Path(tempfile.mkdtemp(prefix="ui-test-data-"))
    data.mkdir(parents=True, exist_ok=True)
    os.environ["LOCAL_AI_DATA_DIR"] = str(data)
    os.environ.setdefault("LOCAL_AI_MODELS_DIR", str(ROOT / "models"))
    sys.path.insert(0, str(ROOT / "backend"))
    os.chdir(ROOT / "backend")
    print(f"UI test backend: data={data}", flush=True)
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
