import pytest
from fastapi import HTTPException

import app


def test_generate_session_id_unique():
    assert app.generate_session_id() != app.generate_session_id()


def test_ensure_services_ready_requires_dependencies(monkeypatch):
    monkeypatch.setattr(app, "config", None, raising=False)
    monkeypatch.setattr(app, "redis_integration", None, raising=False)
    monkeypatch.setattr(app, "bedrock_service", None, raising=False)

    with pytest.raises(HTTPException):
        app.ensure_services_ready()
