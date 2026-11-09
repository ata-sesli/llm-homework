from pathlib import Path

from src.paths import resolve_storage


def test_defaults_to_repo_root(tmp_path):
    assert resolve_storage({}, tmp_path) == tmp_path


def test_reads_dotenv(tmp_path):
    (tmp_path / ".env").write_text("# comment\nTICKET_ROUTER_STORAGE=/Volumes/disk/store\n")
    assert resolve_storage({}, tmp_path) == Path("/Volumes/disk/store")


def test_environment_overrides_dotenv(tmp_path):
    (tmp_path / ".env").write_text("TICKET_ROUTER_STORAGE=/from/dotenv\n")
    assert resolve_storage({"TICKET_ROUTER_STORAGE": "/from/env"}, tmp_path) == Path("/from/env")
