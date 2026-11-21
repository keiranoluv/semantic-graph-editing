"""Keep read-only HDF5 files open within one process, including data workers."""

import os
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np


class ProcessLocalH5:
    """Lazily open one handle per process and never pickle a live HDF5 handle."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._pid = None
        self._arrays = {}

    def get(self) -> h5py.File:
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = pid
        return self._handle

    @contextmanager
    def open(self):
        self.get()
        yield self

    def __getitem__(self, key: str):
        """Map contiguous datasets directly; retain HDF5 access as fallback."""
        h5 = self.get()
        if key not in self._arrays:
            dataset = h5[key]
            offset = dataset.id.get_offset()
            if dataset.chunks is None and dataset.compression is None and offset is not None:
                nbytes = int(dataset.size) * int(dataset.dtype.itemsize)
                if (
                    nbytes > 0
                    and not dataset.dtype.hasobject
                    and offset >= 0
                    and offset + nbytes <= self.path.stat().st_size
                ):
                    self._arrays[key] = np.memmap(
                        self.path, dtype=dataset.dtype, mode="r", offset=offset, shape=dataset.shape
                    )
                else:
                    self._arrays[key] = dataset
            else:
                self._arrays[key] = dataset
        return self._arrays[key]

    def close(self) -> None:
        self._arrays.clear()
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._pid = None

    def __getstate__(self) -> dict:
        return {"path": self.path, "_handle": None, "_pid": None, "_arrays": {}}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
