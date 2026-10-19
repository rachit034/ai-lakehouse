"""Shared MCP server wiring for the lakehouse agents."""
from __future__ import annotations

import os
import sys

_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_AGENTS_DIR)
_MCP_SERVER_DIR = os.path.join(_PROJECT_ROOT, "mcp_server")
_MCP_SERVER_PYTHON = os.path.join(_MCP_SERVER_DIR, ".venv", "bin", "python")

if not os.path.isfile(_MCP_SERVER_PYTHON):
    sys.exit(
        f"MCP server venv not found at {_MCP_SERVER_PYTHON}.\n"
        f"Set it up first: cd {_MCP_SERVER_DIR} && uv venv --python 3.11 .venv "
        f"&& uv pip install --python .venv/bin/python -r requirements.txt"
    )


# Maps to gateway/core/auth.py's roles. Each agent declares which role it
# acts as; the gateway (not this file) is what actually enforces what that
# role can reach — this is just picking the right credential.
_ROLE_ENV_VAR = {
    "data_engineer": "API_KEY_DATA_ENGINEER",
    "data_modeler": "API_KEY_DATA_MODELER",
    "business": "API_KEY_BUSINESS",
}


def lakehouse_mcp_config(agent_name: str, role: str, gateway_url: str | None = None) -> dict:
    """Return the mcp_servers config entry that launches the lakehouse MCP server over stdio.

    agent_name is forwarded as X-Agent-Name on every gateway call the MCP server makes
    for this process, so all of it shows up under one trace tag in MLflow. role selects
    which of the three role-scoped API keys this agent authenticates as.
    """
    if role not in _ROLE_ENV_VAR:
        sys.exit(f"Unknown role {role!r} — must be one of {list(_ROLE_ENV_VAR)}")

    env_var = _ROLE_ENV_VAR[role]
    api_key = os.environ.get(env_var, "")
    if not api_key:
        sys.exit(
            f"{env_var} is not set — every gateway call would 401. "
            f"Export it (same value as .env's {env_var}): "
            f"export {env_var}=$(grep {env_var} ../.env | cut -d= -f2)"
        )

    return {
        "lakehouse": {
            "type": "stdio",
            "command": _MCP_SERVER_PYTHON,
            "args": [os.path.join(_MCP_SERVER_DIR, "server.py")],
            "env": {
                "GATEWAY_URL": gateway_url or os.environ.get("GATEWAY_URL", "http://localhost:8088"),
                "AGENT_NAME": agent_name,
                "GATEWAY_API_KEY": api_key,
            },
        }
    }
