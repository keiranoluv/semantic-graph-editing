"""Prevent concurrent writers to an experiment directory."""

import fcntl
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
