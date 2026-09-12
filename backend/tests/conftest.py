import os
import sys

# Force a lightweight, isolated config *before* anything imports `config` or
# `main` — these are read once at import time via pydantic-settings.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_trace.db")
os.environ.setdefault("ENABLE_GENERATION", "false")  # skip loading Stable Diffusion in CI
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-only")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid

import pytest
from fastapi.testclient import TestClient

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_trace.db")


@pytest.fixture(scope="session", autouse=True)
def _cleanup_test_db():
    yield
    for suffix in ("", "-wal", "-shm"):
        path = TEST_DB_PATH + suffix
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture()
def client():
    from main import app, limiter  # imported here so env vars above are already set

    limiter.reset()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def registered_user(client):
    # Unique per test — the SQLite DB is file-backed and shared across the
    # whole test session, so a fixed email would let one test's password
    # change/reset/delete leak into another test's expectations.
    email = f"pytest-user-{uuid.uuid4().hex[:12]}@example.com"
    password = "correct-horse-battery"
    client.post("/auth/register", json={"email": email, "password": password})
    return {"email": email, "password": password}


@pytest.fixture()
def auth_headers(client, registered_user):
    resp = client.post(
        "/auth/login",
        data={"username": registered_user["email"], "password": registered_user["password"]},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}