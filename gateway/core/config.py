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

    @property
    def role_api_keys(self) -> dict[str, str]:
        return {
            "data_engineer": self.api_key_data_engineer,
            "data_modeler": self.api_key_data_modeler,
            "business": self.api_key_business,
        }

    # Data catalog (DataHub) — optional NL->SQL schema enrichment, not a hard
    # dependency. Empty string disables it outright (datahub_client checks this).
    datahub_gms_url: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
