"""Tests for scripts/publish_db.py.

Publishing to Hugging Face needs a real token and a real dataset repo,
neither of which is available in CI or local dev (see
docs/adr/0006-database-as-published-build-artifact.md). These tests mock
`huggingface_hub.HfApi` and assert the script's own logic: it validates the
artifact and the token before calling out, creates the dataset repo as
public, and uploads the built database under a fixed filename -- never
against a real Hugging Face endpoint.
"""

import sys

import pytest


class FakeHfApi:
    """Records the calls publish() makes, in place of a real HfApi."""

    def __init__(self, token=None):
        self.token = token
        self.create_repo_calls = []
        self.upload_file_calls = []

    def create_repo(self, **kwargs):
        self.create_repo_calls.append(kwargs)

    def upload_file(self, **kwargs):
        self.upload_file_calls.append(kwargs)
        return "https://huggingface.co/datasets/fake/repo/blob/main/names.db"


def test_publish_creates_a_public_dataset_repo_and_uploads_the_database(tmp_path):
    from scripts.publish_db import publish

    db_path = tmp_path / "names.built.db"
    db_path.write_bytes(b"fake sqlite bytes")
    api = FakeHfApi()

    publish("someuser/baby-names-db", str(db_path), token="fake-token", api=api)

    assert len(api.create_repo_calls) == 1
    create_call = api.create_repo_calls[0]
    assert create_call["repo_id"] == "someuser/baby-names-db"
    assert create_call["repo_type"] == "dataset"
    assert create_call["private"] is False
    assert create_call["exist_ok"] is True

    assert len(api.upload_file_calls) == 1
    upload_call = api.upload_file_calls[0]
    assert upload_call["repo_id"] == "someuser/baby-names-db"
    assert upload_call["repo_type"] == "dataset"
    assert upload_call["path_in_repo"] == "names.db"
    assert upload_call["path_or_fileobj"] == str(db_path)


def test_publish_returns_what_the_api_reports(tmp_path):
    from scripts.publish_db import publish

    db_path = tmp_path / "names.built.db"
    db_path.write_bytes(b"fake sqlite bytes")
    api = FakeHfApi()

    result = publish("someuser/baby-names-db", str(db_path), token="fake-token", api=api)

    assert result == "https://huggingface.co/datasets/fake/repo/blob/main/names.db"


def test_publish_refuses_a_missing_database(tmp_path):
    from scripts.publish_db import publish

    api = FakeHfApi()
    with pytest.raises(SystemExit, match="not found"):
        publish("someuser/baby-names-db", str(tmp_path / "missing.db"), token="fake-token", api=api)

    assert api.create_repo_calls == []
    assert api.upload_file_calls == []


def test_publish_refuses_to_run_without_a_token(tmp_path):
    from scripts.publish_db import publish

    db_path = tmp_path / "names.built.db"
    db_path.write_bytes(b"fake sqlite bytes")
    api = FakeHfApi()

    with pytest.raises(SystemExit, match="HF_TOKEN"):
        publish("someuser/baby-names-db", str(db_path), token=None, api=api)

    assert api.create_repo_calls == []


def test_main_reads_repo_id_from_argv_and_token_from_environment(tmp_path, monkeypatch, capsys):
    from scripts import publish_db

    db_path = tmp_path / "names.built.db"
    db_path.write_bytes(b"fake sqlite bytes")
    api = FakeHfApi()

    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(publish_db, "HfApi", lambda token=None: api)
    monkeypatch.setattr(sys, "argv", ["publish_db.py", "someuser/baby-names-db", str(db_path)])

    publish_db.main()

    assert len(api.upload_file_calls) == 1
    out = capsys.readouterr().out
    assert "someuser/baby-names-db" in out
