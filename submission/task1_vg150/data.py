"""Build the VG150 dataset with geometry and optional frozen CLIP features."""

from common.datasets.vg150 import VG150AnnotationLoader, VG150GraphDataset
from common.features.geometry import GeometryFeatureExtractor
from common.features.visual import GeometryVisualFeatureExtractor, VisualFeatureExtractor
from common.graph.builder import SceneGraphBuilder


def make_dataset(root, split, box_policy, visual_cache=None):
    loader = VG150AnnotationLoader(
        root=root,
        split=split,
        split_manifest=root / "splits.json",
        box_policy=box_policy,
    )
    extractor = (
        GeometryVisualFeatureExtractor(VisualFeatureExtractor(visual_cache, root, box_policy))
        if visual_cache is not None
        else GeometryFeatureExtractor()
    )
    return VG150GraphDataset(
        loader=loader,
        feature_extractor=extractor,
        graph_builder=SceneGraphBuilder(),
    )
