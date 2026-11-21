"""CPU tests for held-out cache identity, resume and corruption detection."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from common.datasets.gqa_full import atomic_write, digest, verify_cache
from common.knowledge.transe.core import sha256
from task2_gqa.gqa_test_v1.cache import TestDataset, prepare_cache, test_cache_identity
from task2_gqa.gqa_test_v1.runner import state_hash, summarize


class TestHeldOutCache(unittest.TestCase):
    def test_resume_corruption_and_disjointness(self):
        real = json.loads(Path("data/processed/level2/gqa-v1/manifest.json").read_text())
        label = next(iter(real["vocabulary"]["label_to_idx"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "archives").mkdir()
            (root / "sceneGraphs").mkdir()
            image = root / "101.jpg"
            Image.new("RGB", (20, 20)).save(image)
            with zipfile.ZipFile(root / "archives/images.zip", "w") as archive:
                archive.write(image, "images/101.jpg")
            annotation = root / "sceneGraphs/val_sceneGraphs.json"
            atomic_write(
                annotation,
                {
                    "101": {
                        "width": 20,
                        "height": 20,
                        "objects": {
                            "1": {"name": label, "x": 1, "y": 1, "w": 5, "h": 5, "attributes": [], "relations": []}
                        },
                    }
                },
            )
            manifest = {
                "vocabulary": real["vocabulary"],
                "splits": {"train": ["train"], "val": ["val"], "test": ["101"]},
                "annotation_sha256": {"train_sceneGraphs.json": "unused", "val_sceneGraphs.json": sha256(annotation)},
            }
            atomic_write(root / "manifest.json", manifest)
            (root / "weights").write_bytes(b"fake CPU fixture")
            config = {
                "scope": "full",
                "root": str(root),
                "manifest": str(root / "manifest.json"),
                "weights": str(root / "weights"),
                "cache": str(root / "cache"),
                "cache_shard_size": 1,
                "clip_batch_size": 2,
                "device": "cpu",
            }
            with patch("clip.load") as load, patch(
                "task1_vg150.cache_vg150_regions.extract_vectors", return_value=np.ones((1, 512), dtype=np.float32)
            ):
                load.return_value = (unittest.mock.MagicMock(), None)
                self.assertTrue(prepare_cache(config))
                self.assertEqual(load.call_count, 1)
                self.assertTrue(prepare_cache(config))
                self.assertEqual(load.call_count, 1)
            metadata = verify_cache(root / "cache")
            dataset = TestDataset(root / "cache", metadata)
            try:
                self.assertEqual(len(dataset), 1)
                self.assertEqual(tuple(dataset[0].x.shape), (1, 517))
            finally:
                dataset.close()
            # Recovery after manifest publication was interrupted must not rerun CLIP.
            (root / "cache/manifest.json").unlink()
            with patch("clip.load", side_effect=AssertionError("must reuse shard")):
                self.assertTrue(prepare_cache(config))
            shard = root / "cache/test-00000.h5"
            with shard.open("ab") as f:
                f.write(b"corrupt")
            with self.assertRaises(ValueError):
                verify_cache(root / "cache")
            manifest["splits"]["train"] = ["101"]
            with self.assertRaises(ValueError):
                test_cache_identity(config, manifest)

    def test_dedup_and_summary(self):
        a = {"weight": torch.tensor([1.0]), "feedback.unused": torch.tensor([2.0])}
        self.assertEqual(state_hash(a), state_hash({"weight": torch.tensor([1.0])}))
        self.assertNotEqual(state_hash(a), state_hash({"weight": torch.tensor([2.0])}))
        rows = [
            {
                "model": "fixture",
                "seed": seed,
                "metrics": {
                    mode: {("ng-" if mode == "ng" else "") + key: value for key in ("R@50", "R@100", "mR@50", "mR@100")}
                    for mode in ("gc", "ng")
                },
            }
            for seed, value in [(42, 0.1), (123, 0.2), (3407, 0.3)]
        ]
        result = summarize(rows)["fixture"]["gc"]["mR@50"]
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertAlmostEqual(result["std_sample"], 0.1)
        with self.assertRaises(ValueError):
            digest({"invalid": float("nan")})


if __name__ == "__main__":
    unittest.main()
