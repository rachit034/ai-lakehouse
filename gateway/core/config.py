from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Gravitino (local Glue Catalog substitute) — DuckDB, embedded in this
    # process, attaches its Iceberg REST catalog over HTTP. iceberg_catalog_name
    # must match the catalog name gravitino-init.sh creates in the metalake.
    gravitino_uri: str = "http://gravitino:8090"
    gravitino_metalake: str = "lakehouse"
    iceberg_catalog_name: str = "iceberg"
    default_schema: str = "demo"

    # MinIO / S3
    minio_endpoint: str = "http://minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    lance_uri: str = "s3://lance/"

    # LLM
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "phi3:mini"

    # Cache
    redis_url: str = "redis://redis:6379/0"
    nl_cache_ttl: int = 300  # seconds
    # Cosine distance threshold for the semantic NL->SQL cache — lower = stricter
    # match required. 0.15 tested clean: catches real paraphrases, misses unrelated
    # questions, on the actual product schema.
    nl_cache_distance_threshold: float = 0.15

    # DuckDB — reads run on per-request cursors, at most this many at once
    # (set to 1 to fully serialize, the old behavior). Writes always serialize.
    duckdb_max_concurrent_queries: int = 4
    # Row cap on ad-hoc query responses (/query/sql, /query/nl, /medallion/gold).
    # Hitting it sets truncated=true in the response rather than cutting silently.
    query_max_rows: int = 1000

    # Local-LLM graph Q&A (/graph/nl): how many generate -> EXPLAIN -> fix
    # rounds before giving up, and the row cap injected into generated Cypher.
    graph_nl_max_attempts: int = 3
    graph_nl_row_limit: int = 200

    # Embedding model (fastembed, ONNX — no PyTorch needed)
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Knowledge graph (Neo4j, simulates Amazon Neptune)
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "lakehouse123"

    # Observability (MLflow tracing)
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "ai-lakehouse-gateway"

    # API auth — one key per organizational role, each required with no
    # default (a missing key fails startup loudly rather than silently
    # running with that role unreachable or, worse, open). See core/auth.py
    # for what each role can actually reach.
    api_key_data_engineer: str
    api_key_data_modeler: str
    api_key_business: str
    # Optional named keys, one per person/service, so traces and audits show
    # WHO called rather than only which role: "alice:business:<key>,etl:data_engineer:<key>".
    # The three shared role keys above keep working alongside these.
    api_keys: str = ""

    @property
    def role_api_keys(self) -> dict[str, str]:
        return {
            "data_engineer": self.api_key_data_engineer,
            "data_modeler": self.api_key_data_modeler,
            "business": self.api_key_business,
        }

    @property
    def principal_api_keys(self) -> list[tuple[str, str, str]]:
        """[(principal, role, key)] — shared role keys first, then named keys."""
        out = [(f"{role}-shared", role, key) for role, key in self.role_api_keys.items()]
        for entry in filter(None, (e.strip() for e in self.api_keys.split(","))):
            name, role, key = entry.split(":", 2)
            out.append((name.strip(), role.strip(), key.strip()))
        return out

    # Browser origins allowed to call the gateway. Streamlit calls server-side
    # and Swagger is same-origin, so this only matters for browser apps.
    cors_allow_origins: str = "http://localhost:8501,http://localhost:8088"

    # Data catalog (DataHub) — optional NL->SQL schema enrichment, not a hard
    # dependency. Empty string disables it outright (datahub_client checks this).
    datahub_gms_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
