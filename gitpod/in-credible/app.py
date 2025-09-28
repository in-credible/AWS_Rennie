"""FastAPI application showcasing Redis enhancements for AWS Bedrock Agents."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app_logging import configure_logging
from config import AppConfig
from redis_integration import RedisIntegration
from bedrock_service import BedrockService

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Redis-Enhanced AWS Bedrock Financial Agent",
    description="Demonstration of Redis Enterprise value-add for AWS Bedrock Agent architectures",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


class ChatRequest(BaseModel):
    """Chat request with optional session management."""

    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response with performance metadata."""

    response: str
    cached: bool
    time: float
    session_id: str
    routed: bool = False
    trace_data: Optional[dict] = None


config: Optional[AppConfig] = None
redis_integration: Optional[RedisIntegration] = None
bedrock_service: Optional[BedrockService] = None

total_llm_calls_avoided = 0


def initialize_services() -> None:
    """Initialise configuration, Bedrock clients and Redis integration."""

    global config, redis_integration, bedrock_service

    logger.info("Initialising Redis-enhanced Financial Agent services")
    try:
        config = AppConfig()
        logger.debug("Configuration loaded for region %s", config.bedrock.region)

        bedrock_service = BedrockService(config.bedrock)
        redis_integration = RedisIntegration(config, bedrock_service.bedrock_runtime)

        logger.info("Services initialised successfully")
        logger.info(
            "Comparing redis-enhanced agent %s against standard agent %s",
            config.bedrock.agent_id_redis,
            config.bedrock.agent_id_opensearch,
        )
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Failed to initialise services: %s", exc)
        raise


def ensure_services_ready() -> None:
    """Raise an HTTPException if services failed to start."""

    if not (config and redis_integration and bedrock_service):
        logger.error("Service dependencies are not ready")
        raise HTTPException(status_code=503, detail="Services are initialising, try again shortly")


def generate_session_id() -> str:
    """Generate a unique session identifier."""

    return str(uuid.uuid4())


def _log_background_exception(task: asyncio.Task) -> None:
    """Centralised handling for background task failures."""

    if task.cancelled():
        logger.warning("Background task was cancelled")
        return

    exc = task.exception()
    if exc:
        logger.error("Background task failed: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise dependencies when the FastAPI application starts."""

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, initialize_services)


@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve the static comparison UI."""

    return FileResponse("index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/chat/arch1", response_model=ChatResponse)
async def redis_enhanced_architecture(request: ChatRequest) -> ChatResponse:
    """Redis-enhanced Bedrock architecture with caching, memory and routing."""

    ensure_services_ready()

    global total_llm_calls_avoided

    start_time = time.perf_counter()
    timing_log: Dict[str, float] = {}

    session_id = request.session_id or generate_session_id()
    logger.info("[REDIS-ARCH] Processing query for session %s", session_id[:8])

    timing_log["session_init"] = round(time.perf_counter() - start_time, 3)

    try:
        loop = asyncio.get_running_loop()

        routing_start = time.perf_counter()
        route_decision, route_confidence = redis_integration.route_query(request.query)
        timing_log["routing"] = round(time.perf_counter() - routing_start, 3)

        if route_decision == "not_allowed":
            total_llm_calls_avoided += 1
            elapsed = round(time.perf_counter() - start_time, 3)
            logger.info("[REDIS-ARCH] Query filtered (confidence %.3f)", route_confidence)
            return ChatResponse(
                response="Please ask me a question related to financial analysis, investment research, or personal information for memory testing.",
                cached=False,
                time=elapsed,
                routed=True,
                session_id=session_id,
                trace_data={
                    "type": "filtered",
                    "confidence": round(route_confidence, 3),
                    "reason": "Query identified as off-topic",
                    "allowed_topics": [
                        "Financial analysis",
                        "Investment research",
                        "Company performance",
                        "Market trends",
                        "Personal information",
                    ],
                    "llm_call_avoided": True,
                },
            )

        logger.debug("[REDIS-ARCH] Query allowed (confidence %.3f)", route_confidence)

        embedding_start = time.perf_counter()
        query_embedding = await loop.run_in_executor(
            None, redis_integration.embedding_service.embed, request.query
        )
        timing_log["embedding_generation"] = round(time.perf_counter() - embedding_start, 3)

        parallel_start = time.perf_counter()
        memory_task = asyncio.create_task(
            redis_integration.search_conversation_memory(
                session_id, request.query, query_embedding=query_embedding
            )
        )
        cache_task = asyncio.create_task(
            redis_integration.search_semantic_cache(request.query, query_embedding=query_embedding)
        )
        memories, cache_result = await asyncio.gather(memory_task, cache_task)
        timing_log["parallel_search"] = round(time.perf_counter() - parallel_start, 3)

        memory_context = redis_integration.format_memory_context(memories)
        if memories:
            logger.debug("[REDIS-ARCH] Retrieved %s memories", len(memories))

        if cache_result:
            total_llm_calls_avoided += 1
            elapsed = round(time.perf_counter() - start_time, 3)
            logger.info(
                "[REDIS-ARCH] Cache hit (similarity %.3f). LLM calls avoided=%s",
                cache_result.similarity_score,
                total_llm_calls_avoided,
            )
            return ChatResponse(
                response=cache_result.response,
                cached=True,
                time=elapsed,
                session_id=session_id,
                trace_data={
                    "type": "cache_hit",
                    "similarity_score": round(cache_result.similarity_score, 3),
                    "matched_query": cache_result.query,
                    "cache_timestamp": cache_result.timestamp,
                    "memory_used": len(memories) if memories else 0,
                    "timing_breakdown": timing_log,
                    "llm_call_avoided": True,
                },
            )

        bedrock_start = time.perf_counter()
        response_text = await bedrock_service.invoke_redis_enhanced_agent(
            query=request.query,
            context=memory_context,
        )
        timing_log["bedrock_call"] = round(time.perf_counter() - bedrock_start, 3)
        logger.info("[REDIS-ARCH] Agent response in %.3fs", timing_log["bedrock_call"])

        async def persist_results() -> None:
            await redis_integration.store_in_cache(request.query, response_text)
            await redis_integration.store_conversation_memory(
                session_id, request.query, response_text
            )

        background_start = time.perf_counter()
        background_task = asyncio.create_task(persist_results())
        background_task.add_done_callback(_log_background_exception)
        timing_log["background_task_launch"] = round(time.perf_counter() - background_start, 3)

        total_time = round(time.perf_counter() - start_time, 3)
        logger.info(
            "[REDIS-ARCH] Completed in %.3fs (routing=%.3fs, embedding=%.3fs, search=%.3fs, bedrock=%.3fs)",
            total_time,
            timing_log.get("routing", 0.0),
            timing_log.get("embedding_generation", 0.0),
            timing_log.get("parallel_search", 0.0),
            timing_log.get("bedrock_call", 0.0),
        )

        memory_trace = []
        for mem in memories[:3]:
            memory_trace.append(
                {
                    "user_said": mem.user_message,
                    "agent_responded": (
                        mem.bot_response[:100] + "..." if len(mem.bot_response) > 100 else mem.bot_response
                    ),
                    "relevance": round(mem.similarity, 3),
                    "when": mem.timestamp,
                }
            )

        return ChatResponse(
            response=response_text,
            cached=False,
            time=total_time,
            session_id=session_id,
            trace_data={
                "type": "fresh_query",
                "bedrock_time": timing_log.get("bedrock_call", 0.0),
                "preprocessing_time": round(bedrock_start - start_time, 3),
                "memory_context_used": memory_trace,
                "route_confidence": round(route_confidence, 3),
                "query_intent": redis_integration.intent_classifier.classify_intent(request.query),
                "timing_breakdown": timing_log,
            },
        )

    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.exception("[REDIS-ARCH] Request failed: %s", exc)
        return ChatResponse(
            response=(
                "Redis-Enhanced Architecture Error: "
                f"{exc}\n\n(Redis components are running, the failure likely originates from Bedrock)"
            ),
            cached=False,
            time=elapsed,
            session_id=session_id,
        )


@app.post("/chat/arch2", response_model=ChatResponse)
async def standard_aws_architecture(request: ChatRequest) -> ChatResponse:
    """Baseline AWS architecture without Redis optimisations."""

    ensure_services_ready()

    start_time = time.perf_counter()
    session_id = request.session_id or generate_session_id()
    logger.info("[STANDARD-ARCH] Processing query for session %s", session_id[:8])

    try:
        bedrock_start = time.perf_counter()
        response_text = await bedrock_service.invoke_standard_agent(request.query)
        bedrock_elapsed = round(time.perf_counter() - bedrock_start, 3)
        total_time = round(time.perf_counter() - start_time, 3)

        logger.info(
            "[STANDARD-ARCH] Response in %.3fs (total %.3fs)",
            bedrock_elapsed,
            total_time,
        )

        return ChatResponse(
            response=response_text,
            cached=False,
            time=total_time,
            session_id=session_id,
        )

    except Exception as exc:
        elapsed = round(time.perf_counter() - start_time, 3)
        logger.exception("[STANDARD-ARCH] Request failed: %s", exc)
        return ChatResponse(
            response=f"Standard Architecture Error: {exc}",
            cached=False,
            time=elapsed,
            session_id=session_id,
        )


@app.post("/chat", response_model=ChatResponse)
async def legacy_chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Backward compatible endpoint that maps to the redis-enhanced flow."""

    return await redis_enhanced_architecture(request)


@app.post("/clear-cache")
async def clear_semantic_cache() -> Dict[str, int]:
    """Clear Redis semantic cache for testing."""

    ensure_services_ready()

    try:
        result = redis_integration.clear_cache()
        logger.info("[MANAGEMENT] Cleared %s cache entries", result["cache_cleared"])
        return result
    except Exception as exc:  # pragma: no cover - depends on Redis deployment
        logger.exception("[MANAGEMENT] Failed to clear cache: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {exc}")


@app.post("/clear-memory")
async def clear_conversation_memory() -> Dict[str, int]:
    """Clear Redis conversation memory for testing."""

    ensure_services_ready()

    try:
        result = redis_integration.clear_memory()
        logger.info("[MANAGEMENT] Cleared %s memory entries", result["memory_cleared"])
        return result
    except Exception as exc:  # pragma: no cover - depends on Redis deployment
        logger.exception("[MANAGEMENT] Failed to clear memory: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {exc}")


@app.post("/reset-all")
async def reset_all_redis_data() -> Dict[str, int]:
    """Clear both cache and memory for a full demo reset."""

    ensure_services_ready()

    try:
        cache_result = redis_integration.clear_cache()
        memory_result = redis_integration.clear_memory()

        total_cleared = cache_result["cache_cleared"] + memory_result["memory_cleared"]
        logger.info(
            "[MANAGEMENT] Reset complete (cache=%s, memory=%s)",
            cache_result["cache_cleared"],
            memory_result["memory_cleared"],
        )

        return {
            "message": "Redis cache and conversation memory cleared",
            "cache_cleared": cache_result["cache_cleared"],
            "memory_cleared": memory_result["memory_cleared"],
            "total_cleared": total_cleared,
        }
    except Exception as exc:  # pragma: no cover - depends on Redis deployment
        logger.exception("[MANAGEMENT] Failed to reset data: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to reset all data: {exc}")


@app.get("/health")
async def health_check() -> Dict[str, object]:
    """Provide a combined health snapshot for Redis and Bedrock services."""

    ensure_services_ready()

    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "redis": redis_integration.get_stats(),
            "bedrock": bedrock_service.health_check(),
            "config": {
                "cache_threshold": config.cache.similarity_threshold,
                "memory_threshold": config.cache.memory_similarity_threshold,
                "cache_ttl_hours": config.cache.cache_ttl_seconds // 3600,
                "memory_ttl_days": config.cache.memory_ttl_seconds // (3600 * 24),
            },
        }
    except Exception as exc:
        logger.exception("[HEALTH] Health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/stats")
async def get_redis_statistics() -> Dict[str, object]:
    """Return Redis statistics and optimisation impact."""

    ensure_services_ready()

    try:
        return {
            "redis_stats": redis_integration.get_stats(),
            "config": {
                "similarity_threshold": config.cache.similarity_threshold,
                "memory_threshold": config.cache.memory_similarity_threshold,
                "cache_ttl": config.cache.cache_ttl_seconds,
                "memory_ttl": config.cache.memory_ttl_seconds,
            },
            "performance_impact": {
                "llm_calls_avoided": total_llm_calls_avoided,
                "optimisations": ["Semantic Caching", "Query Routing", "Conversation Memory"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:  # pragma: no cover - depends on Redis deployment
        logger.exception("[STATS] Failed to gather statistics: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {exc}")


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
