"""Immutable evidence files and durable per-sample attempt logs."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from tools.prepare_magicbrush import digest, encode


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encode(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable_evidence_changed:{path}; use a new version")
        return
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another_process_holds:{directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def sample_dir(output, identifier):
    return Path(output) / "samples" / digest(identifier)[:24]


def event(path, payload):
    # Per-run lock prevents concurrent writers. Raw attempts use atomic files;
    # events are diagnostic only, so a torn final JSONL line never loses work.
    with Path(path).open("ab") as stream:
        stream.write((json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
