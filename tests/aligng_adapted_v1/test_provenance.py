import ast
import hashlib
import json
from pathlib import Path


def test_frozen_source_inventory_and_reference_hashes_unchanged():
    root = Path(__file__).resolve().parents[2]
    provenance = json.loads((root / "experiments/aligng_adapted_v1/provenance.json").read_text())
    expected = provenance["historical_code_sha256"]
    actual_src = {str(p.relative_to(root)) for p in (root / "src").rglob("*.py")}
    assert actual_src == {p for p in expected if p.startswith("src/")}
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    assert (
        hashlib.sha256((root / "tests/aligng_adapted_v1/upstream_reference.py").read_bytes()).hexdigest()
        == provenance["reference_fixture_sha256"]
    )
    assert (
        hashlib.sha256((root / "experiments/aligng_adapted_v1/LICENSE-AlignG.txt").read_bytes()).hexdigest()
        == provenance["upstream_license_sha256"]
    )
    text = (root / "tests/aligng_adapted_v1/upstream_reference.py").read_text()
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "SGGLayer")
    excerpt = "".join(text.splitlines(keepends=True)[cls.lineno - 1 : cls.end_lineno])
    assert hashlib.sha256(excerpt.encode()).hexdigest() == provenance["sgg_layer_excerpt_sha256"]
