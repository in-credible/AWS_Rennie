import os

import pytest

from config import AppConfig


REQUIRED_ENV_VARS = {
    "REDIS_HOST": "localhost",
    "REDIS_PASSWORD": "password",
    "BEDROCK_AGENT_ID_REDIS": "agent-redis",
    "BEDROCK_AGENT_ID_OPENSEARCH": "agent-opensearch",
}


def test_config_requires_environment(monkeypatch):
    for key in REQUIRED_ENV_VARS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ValueError):
        AppConfig()


def test_config_defaults(monkeypatch):
    for key, value in REQUIRED_ENV_VARS.items():
        monkeypatch.setenv(key, value)

    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")

    cfg = AppConfig()

    assert cfg.redis.host == "localhost"
    assert cfg.redis.port == 6380
    assert cfg.cache.similarity_threshold == 0.8
    assert cfg.bedrock.region == "us-west-2"
