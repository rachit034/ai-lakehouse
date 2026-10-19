"""
Role-based API-key auth for the gateway.

Three organizational roles, one key each — not one shared key for everyone:
  - data_engineer: owns landing data (/ingest/*)
  - data_modeler:  owns designing/committing the entity model
                    (/graph/build, /graph/infer*, /medallion/silver/*, gold materialization)
  - business:      the query surface — /query/*, /graph/query, /graph/stats,
                    read-only gold queries. Every role can also reach this
                    surface, since verifying your own ingested/modeled data
                    is a legitimate need for all three, not just business.

A key only maps to one role; require_role() 403s if that role isn't listed
as allowed for the route, so e.g. a business key can't call /ingest/csv or
/graph/infer/materialize even though it can authenticate successfully.

In production swap this for real IAM/OAuth with actual per-user identities;
this is the minimum viable *role* boundary for a prototype that, before this,
had one flat credential shared by everyone for everything.
"""
from __future__ import annotations

import hmac
from typing import Literal

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from core.config import settings

Role = Literal["data_engineer", "data_modeler", "business"]

ALL_ROLES: tuple[Role, ...] = ("data_engineer", "data_modeler", "business")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _role_for_key(api_key: str) -> Role | None:
    for role, key in settings.role_api_keys.items():
        if key and hmac.compare_digest(api_key, key):
            return role  # type: ignore[return-value]
    return None


def require_role(*allowed: Role):
    """
    FastAPI dependency factory. Authenticates X-API-Key, resolves which role
    it belongs to, and 403s if that role isn't in `allowed`. Stashes the
    resolved role on request.state.role so route handlers can use it for
    role-aware behavior (e.g. PII redaction bypass for engineer/modeler)
    without re-deriving it.
    """

    def _dep(request: Request, api_key: str | None = Security(_api_key_header)) -> Role:
        if not api_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing X-API-Key header")
        role = _role_for_key(api_key)
        if role is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid X-API-Key")
        if role not in allowed:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Role '{role}' cannot call this endpoint — requires one of {list(allowed)}",
            )
        request.state.role = role
        return role

    return _dep
