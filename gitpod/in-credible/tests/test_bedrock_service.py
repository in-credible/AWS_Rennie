import json
from typing import Dict, Iterator
from unittest.mock import MagicMock

import pytest

from bedrock_service import BedrockService
from config import BedrockConfig


@pytest.fixture()
def bedrock_config() -> BedrockConfig:
    return BedrockConfig(
        region="us-west-2",
        agent_id_redis="redis-agent",
        agent_id_opensearch="standard-agent",
    )


@pytest.fixture()
def runtime_client() -> MagicMock:
    mock = MagicMock()
    body = MagicMock()
    embedding = [0.0] * 1536
    body.read.return_value = json.dumps({"embedding": embedding}).encode()
    mock.invoke_model.return_value = {"body": body}
    return mock


@pytest.fixture()
def agent_runtime_client() -> MagicMock:
    mock = MagicMock()

    def completion_stream() -> Iterator[Dict[str, Dict[str, bytes]]]:
        yield {"chunk": {"bytes": b"Hello"}}
        yield {"chunk": {"bytes": b" world"}}

    mock.invoke_agent.return_value = {"completion": completion_stream()}
    return mock


@pytest.mark.asyncio()
async def test_invoke_standard_agent(bedrock_config, runtime_client, agent_runtime_client):
    service = BedrockService(
        bedrock_config,
        runtime_client=runtime_client,
        agent_runtime_client=agent_runtime_client,
    )

    result = await service.invoke_standard_agent("How is revenue?")

    assert result == "Hello world"
    agent_runtime_client.invoke_agent.assert_called_once()


@pytest.mark.asyncio()
async def test_invoke_redis_agent_with_context(bedrock_config, runtime_client, agent_runtime_client):
    service = BedrockService(
        bedrock_config,
        runtime_client=runtime_client,
        agent_runtime_client=agent_runtime_client,
    )

    result = await service.invoke_redis_enhanced_agent(
        "Question?", context="User said something"
    )

    assert "Hello world" in result


def test_generate_embedding_validates_dimension(bedrock_config, runtime_client, agent_runtime_client):
    service = BedrockService(
        bedrock_config,
        runtime_client=runtime_client,
        agent_runtime_client=agent_runtime_client,
    )

    embedding = service.generate_embedding("hello")

    assert len(embedding) == bedrock_config.embedding_dimensions


def test_process_streaming_response_raises_without_data(bedrock_config, runtime_client, agent_runtime_client):
    service = BedrockService(
        bedrock_config,
        runtime_client=runtime_client,
        agent_runtime_client=agent_runtime_client,
    )

    with pytest.raises(Exception):
        service._process_streaming_response({})
