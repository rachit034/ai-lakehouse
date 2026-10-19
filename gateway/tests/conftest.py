"""Unit tests run with no services up: settings get dummy keys, and anything
that needs DuckDB uses a local in-memory database instead of Gravitino/MinIO."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("API_KEY_DATA_ENGINEER", "test-engineer-key")
os.environ.setdefault("API_KEY_DATA_MODELER", "test-modeler-key")
os.environ.setdefault("API_KEY_BUSINESS", "test-business-key")
os.environ.setdefault("API_KEYS", "alice:business:alice-key,etl:data_engineer:etl-key")
os.environ.setdefault("DATAHUB_GMS_URL", "")
