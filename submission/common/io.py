"""Immutable JSON artifacts and streaming SHA-256 checks."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed(path, content):
    """Create atomically; identical resume is allowed, changed content is refused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable_output_changed:{path}")
        return
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        try:
            os.link(name, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"immutable_output_changed:{path}") from None
    finally:
        os.unlink(name)


def write(path, value):
    fixed(path, encode(value))


def read(path):
    return json.loads(Path(path).read_text())
