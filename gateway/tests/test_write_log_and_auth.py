from core import auth
from core.config import settings
from services.write_log import is_stale


def test_stale_when_written_after_build():
    assert is_stale("2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00") is True


def test_fresh_when_built_after_write():
    assert is_stale("2026-01-03T00:00:00Z", "2026-01-02T00:00:00+00:00") is False


def test_unknown_when_never_built():
    assert is_stale(None, "2026-01-02T00:00:00+00:00") is None


def test_named_keys_resolve_to_principal_and_role():
    assert auth._principal_for_key("alice-key") == ("alice", "business")
    assert auth._principal_for_key("etl-key") == ("etl", "data_engineer")
    assert auth._principal_for_key(settings.api_key_data_modeler) == ("data_modeler-shared", "data_modeler")
    assert auth._principal_for_key("nope") is None
