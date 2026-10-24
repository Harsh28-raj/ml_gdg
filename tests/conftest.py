import os
import tempfile

# Isolated embedding cache per test session (must be set before app import).
os.environ.setdefault("ML_CACHE_DB", os.path.join(tempfile.mkdtemp(prefix="mlcache_"), "cache.sqlite3"))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """One client per session so models load only once."""
    with TestClient(app) as c:
        yield c
