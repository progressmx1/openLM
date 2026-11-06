"""Root pytest config: isolate the test run from the real ``data/`` folder.

``app.main`` builds an application (SQLite DB, ``backend.log``) at import time, and
``app.core.paths`` reads ``LOCAL_AI_DATA_DIR`` when first imported. This file is loaded
before any test module imports ``app``, so the whole run uses a throwaway directory.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_DATA_DIR = tempfile.mkdtemp(prefix="local-ai-test-data-")
os.environ["LOCAL_AI_DATA_DIR"] = _DATA_DIR
os.environ["LOCAL_AI_MODELS_DIR"] = os.path.join(_DATA_DIR, "models")
os.environ["LOCAL_AI_DOCUMENTS_DIR"] = os.path.join(_DATA_DIR, "documents")
atexit.register(shutil.rmtree, _DATA_DIR, True)
