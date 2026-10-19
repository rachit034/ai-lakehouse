"""
MLflow tracing for the gateway. In production point MLFLOW_TRACKING_URI at a
managed MLflow (Databricks) or self-hosted tracking server instead.

Every HTTP request gets one top-level trace (named "<METHOD> <path>"); service
functions decorated with @mlflow.trace nest inside it automatically since they
run within the request's active span context. Agents/MCP tool calls forward
X-Agent-Name and X-Run-Id headers so every request one agent invocation made
can be found via those trace tags in the MLflow UI.
"""
from __future__ import annotations

import mlflow
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from core.config import settings


def configure_tracing() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip noisy/uninteresting paths
        if request.url.path in ("/health", "/ready", "/docs", "/openapi.json"):
            return await call_next(request)

        with mlflow.start_span(name=f"{request.method} {request.url.path}") as span:
            tags = {}
            agent_name = request.headers.get("x-agent-name")
            run_id = request.headers.get("x-run-id")
            if agent_name:
                tags["agent_name"] = agent_name
            if run_id:
                tags["agent_run_id"] = run_id
            if tags:
                mlflow.update_current_trace(tags=tags)

            span.set_inputs({"method": request.method, "path": request.url.path})
            response = await call_next(request)
            span.set_outputs({"status_code": response.status_code})
            return response
