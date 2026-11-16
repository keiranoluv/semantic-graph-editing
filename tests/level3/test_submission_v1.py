import hashlib
import json
import zipfile

import pytest

from tools.level3_submission_v1 import bundle, demo


def archive(tmp_path, name="payload.txt", payload=b"frozen evidence", expected=None, extra=None):
    path = tmp_path / "bundle.zip"
    index = {
        "commit": "fixture",
        "counts": {"sources": 1},
        "files": {name: {"size": len(payload), "sha256": expected or hashlib.sha256(payload).hexdigest()}},
    }
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, payload)
        z.writestr("BACKUP_INDEX.json", json.dumps(index))
        if extra:
            z.writestr(extra, b"unlisted")
    path.with_suffix(".zip.sha256").write_text(bundle.sha(path) + "  bundle.zip\n")
    return path


def test_archive_checks_every_member_and_commit(tmp_path):
    result = bundle.verify(archive(tmp_path))
    assert result["files_verified"] == 1 and result["code_commit"] == "fixture"
    assert result["counts"] == {"sources": 1}


def test_archive_detects_outer_corruption(tmp_path):
    path = archive(tmp_path)
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="archive_checksum_mismatch"):
        bundle.verify(path)


def test_archive_detects_wrong_inner_hash(tmp_path):
    with pytest.raises(ValueError, match="archive_member_changed"):
        bundle.verify(archive(tmp_path, expected="wrong"))


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "nested/../../escape", "nested\\escape"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe_archive_path"):
        bundle.verify(archive(tmp_path, name=name))


def test_archive_rejects_unlisted_payload(tmp_path):
    with pytest.raises(ValueError, match="archive_membership_mismatch"):
        bundle.verify(archive(tmp_path, extra="surprise.txt"))


def test_portable_demo_resolves_relative_path_and_refuses_corruption(tmp_path):
    file = tmp_path / "evidence.json"
    file.write_text('{"label":true}')
    manifest = {"files": {"evidence.json": {"sha256": bundle.sha(file)}}}
    assert demo.checked(tmp_path, manifest, "evidence.json") == file
    file.write_text('{"label":false}')
    with pytest.raises(ValueError, match="demo_artifact_changed"):
        demo.checked(tmp_path, manifest, "evidence.json")


def test_graph_html_keeps_direction_and_escapes_labels():
    graph = {
        "nodes": [{"id": "a", "label": "<script>"}, {"id": "b", "label": "boat"}],
        "edges": [{"id": "e", "subject": "b", "object": "a", "predicate": "on"}],
    }
    rendered = demo.graph_html(graph, "Directed graph")
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert rendered.index("b: boat") < rendered.index("a: &lt;script&gt;")
