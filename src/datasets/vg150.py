"""VG150 annotation adapter and PyG dataset interface (draft only)."""

from pathlib import Path
from typing import Literal

from torch_geometric.data import Data, Dataset

from src.datasets.schemas import SceneGraphRecord

Split = Literal["train", "val", "test"]


class VG150AnnotationLoader:
    """Read a pinned VG150 annotation format into normalized scene records.

    A real parser will be added *after* selecting the VG150 source variant.
    No implicit random splits or train/test statistics are allowed here.
    """

    def __init__(self, root: Path, split: Split, split_manifest: Path) -> None:
        self.root = Path(root)
        self.split = split
        self.split_manifest = Path(split_manifest)

    def load(self, index: int) -> SceneGraphRecord:
        """Return one normalized image record from the selected split."""
        raise NotImplementedError("VG150 source format not selected yet")

    def __len__(self) -> int:
        """Number of images in the selected split."""
        raise NotImplementedError("VG150 split manifest not loaded yet")


class VG150GraphDataset(Dataset):
    """PyG adapter; graph creation will be delegated to a graph builder.

    TODO: dependency injection of feature extractor and graph builder.
    """

    def __init__(self, loader: VG150AnnotationLoader) -> None:
        super().__init__(root=None)
        self.loader = loader

    def len(self) -> int:
        """Number of graph samples (images), not objects or relationships."""
        return len(self.loader)

    def get(self, idx: int) -> Data:
        """Return one PyG graph; not implemented in the skeleton phase."""
        raise NotImplementedError("Connect loader, features, and graph builder")
