"""
AWS Bedrock Agent service functions.
"""

import json
import time
import logging
import asyncio
from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from config import BedrockConfig


AGENT_CALL_TIMEOUT = 45.0  # seconds


logger = logging.getLogger(__name__)


class BedrockService:
    """Encapsulates all AWS Bedrock Agent interactions."""

    def __init__(
        self,
        config: BedrockConfig,
        *,
        runtime_client: Optional[object] = None,
        agent_runtime_client: Optional[object] = None,
    ) -> None:
        self.config = config
        self.bedrock_runtime = runtime_client or boto3.client(
            "bedrock-runtime", region_name=config.region
        )
        self.bedrock_agent_runtime = agent_runtime_client or boto3.client(
            "bedrock-agent-runtime", region_name=config.region
        )

        logger.info("[BEDROCK] Service initialized for region: %s", config.region)
        logger.debug(
            "[BEDROCK] Using agents redis=%s standard=%s",
            config.agent_id_redis,
            config.agent_id_opensearch,
        )
    
    async def invoke_redis_enhanced_agent(self, query: str, context: str = "") -> str:
        """
        Invoke the Redis-enhanced Bedrock Agent.
        
        This agent has access to Redis Vector Database for enhanced
        knowledge retrieval and caching capabilities.
        
        Args:
            query: User query
            context: Additional context from conversation memory
            
        Returns:
            Agent response text
        """
        return await self._invoke_agent(
            agent_id=self.config.agent_id_redis,
            query=query,
            context=context,
            session_prefix="redis-arch"
        )
    
    async def invoke_standard_agent(self, query: str) -> str:
        """
        Invoke the standard AWS Bedrock Agent.
        
        This agent uses OpenSearch for knowledge retrieval,
        representing the baseline AWS architecture.
        
        Args:
            query: User query
            
        Returns:
            Agent response text
        """
        return await self._invoke_agent(
            agent_id=self.config.agent_id_opensearch,
            query=query,
            context="",
            session_prefix="standard-arch"
        )
    
    async def _invoke_agent(self, agent_id: str, query: str, 
                          context: str, session_prefix: str) -> str:
        """
        Core agent invocation with comprehensive error handling.
        
        Args:
            agent_id: Bedrock Agent ID
            query: User query
            context: Additional context to include
            session_prefix: Session identifier prefix
            
        Returns:
            Complete agent response
            
        Raises:
            Exception: For unrecoverable agent errors
        """
        start_time = time.time()
        
        try:
            enhanced_query = self._prepare_query(query, context)
            session_id = f"session-{session_prefix}-{int(time.time())}"

            logger.debug(
                "[BEDROCK] Invoking agent %s with session %s (chars=%s)",
                agent_id,
                session_id,
                len(enhanced_query),
            )

            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._invoke_agent_sync,
                    agent_id,
                    enhanced_query,
                    session_id,
                ),
                timeout=AGENT_CALL_TIMEOUT,
            )

            elapsed = time.time() - start_time
            logger.info(
                "[BEDROCK] Response received in %.3fs (length=%s chars)",
                elapsed,
                len(response),
            )

            return response

        except asyncio.TimeoutError as exc:
            elapsed = time.time() - start_time
            error_msg = (
                f"Agent {agent_id} timed out after {elapsed:.3f}s with timeout {AGENT_CALL_TIMEOUT}s"
            )
            logger.error(error_msg)
            raise Exception(error_msg) from exc

        except Exception as exc:  # pragma: no cover - defensive logging
            elapsed = time.time() - start_time
            error_msg = f"Agent {agent_id} failed after {elapsed:.3f}s: {exc}"
            logger.error(error_msg)
            raise Exception(error_msg) from exc
    
    def _prepare_query(self, query: str, context: str) -> str:
        """
        Prepare enhanced query with conversation context.
        
        Args:
            query: Original user query
            context: Conversation context from memory
            
        Returns:
            Enhanced query string
        """
        if not context:
            return query
        
        return f"{context}\n\nCurrent question: {query}"
    
    def _invoke_agent_sync(self, agent_id: str, query: str, session_id: str) -> str:
        """
        Synchronous agent invocation for thread pool execution.
        
        Args:
            agent_id: Bedrock Agent ID
            query: Enhanced query text
            session_id: Unique session identifier
            
        Returns:
            Complete agent response text
            
        Raises:
            Various Bedrock-related exceptions
        """
        try:
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=self.config.agent_alias_id,
                sessionId=session_id,
                inputText=query
            )
            
            return self._process_streaming_response(response)
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(
                "[BEDROCK] ClientError during invoke [%s]: %s", error_code, error_message
            )
            raise Exception(f"Bedrock ClientError [{error_code}]: {error_message}")
            
        except BotoCoreError as e:
            logger.error("[BEDROCK] BotoCoreError during invoke: %s", e)
            raise Exception(f"Bedrock BotoCoreError: {str(e)}") from e
            
        except Exception as e:
            logger.error("[BEDROCK] Unexpected invoke error: %s", e)
            raise Exception(f"Unexpected Bedrock error: {str(e)}") from e
    
    def _process_streaming_response(self, response: dict) -> str:
        """
        Process Bedrock Agent streaming response.
        
        Args:
            response: Raw Bedrock response with event stream
            
        Returns:
            Complete response text
            
        Raises:
            Exception: If response processing fails
        """
        try:
            event_stream = response.get("completion")
            if not event_stream:
                raise Exception("No completion stream in Bedrock response")
            
            full_response = ""
            chunk_count = 0
            
            for event in event_stream:
                if "chunk" in event:
                    chunk = event["chunk"]["bytes"].decode("utf-8")
                    full_response += chunk
                    chunk_count += 1
                    
                elif "trace" in event:
                    # Log trace information for debugging
                    trace = event.get("trace", {})
                    trace_type = trace.get("type", "unknown")
                    if trace_type != "guardrailTrace":  # Reduce noise
                        logger.debug("[BEDROCK] Trace event: %s", trace_type)
            
            if not full_response:
                raise Exception("Empty response from Bedrock agent")
            
            logger.debug("[BEDROCK] Processed %s response chunks", chunk_count)
            return full_response.strip()
            
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[BEDROCK] Response processing failed: %s", e)
            raise Exception(f"Response processing failed: {str(e)}") from e
    
    def generate_embedding(self, text: str) -> list:
        """
        Generate embedding using Amazon Titan Embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.debug("[BEDROCK] Generating embedding (chars=%s)", len(text))
            response = self.bedrock_runtime.invoke_model(
                modelId=self.config.embedding_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            
            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]
            
            if len(embedding) != self.config.embedding_dimensions:
                raise Exception(
                    f"Unexpected embedding dimensions: {len(embedding)} "
                    f"(expected {self.config.embedding_dimensions})"
                )
            
            return embedding
            
        except ClientError as e:  # pragma: no cover - depends on AWS response
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error("[BEDROCK] Embedding generation failed [%s]: %s", error_code, e)
            raise Exception(f"Embedding generation failed [{error_code}]: {str(e)}") from e

        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[BEDROCK] Embedding error: %s", e)
            raise Exception(f"Embedding error: {str(e)}") from e
    
    def health_check(self) -> dict:
        """
        Perform health check on Bedrock services.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            "bedrock_runtime": False,
            "bedrock_agent_runtime": False,
            "embedding_service": False
        }
        
        # Test Bedrock Runtime (embeddings)
        try:
            self.generate_embedding("test")
            health_status["bedrock_runtime"] = True
            health_status["embedding_service"] = True
        except Exception as e:
            logger.warning("[BEDROCK] Runtime health check failed: %s", e)
        
        # Test Agent Runtime connectivity (without full invocation)
        try:
            # This is a minimal check - just verify the client is configured
            self.bedrock_agent_runtime.meta.client
            health_status["bedrock_agent_runtime"] = True
        except Exception as e:
            logger.warning("[BEDROCK] Agent runtime health check failed: %s", e)
        
        return health_status
