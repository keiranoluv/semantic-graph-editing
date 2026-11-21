import hashlib
import io

import pytest

from extension_pvsg.m2 import download


class Response(io.BytesIO):
    status = 200
    headers = {}


def test_download_checks_identity_and_reuses_verified_file(tmp_path, monkeypatch):
    payload = b"small fake archive"
    item = {
        "filename": "sample.zip",
        "url": "https://example.invalid/fake",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    assert not download.download_one(item, tmp_path)["cached"]
    assert download.download_one(item, tmp_path)["cached"]
    (tmp_path / "sample.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="failed verification"):
        download.download_one(item, tmp_path)


def test_server_ignoring_range_restarts_partial_safely(tmp_path, monkeypatch):
    payload = b"complete body"
    item = {
        "filename": "sample.zip",
        "url": "https://example.invalid/fake",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (tmp_path / "sample.zip.partial").write_bytes(payload[:3])
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *args, **kwargs: Response(payload))
    download.download_one(item, tmp_path)
    assert (tmp_path / "sample.zip").read_bytes() == payload


def test_bad_hash_never_promoted(tmp_path, monkeypatch):
    item = {"filename": "sample.zip", "url": "https://example.invalid/fake", "bytes": 4, "sha256": "0" * 64}
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"fake"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download.download_one(item, tmp_path)
    assert not (tmp_path / "sample.zip").exists()
