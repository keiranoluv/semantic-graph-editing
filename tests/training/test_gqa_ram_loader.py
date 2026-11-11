"""CPU-worker graph preload must preserve inputs and exact resume behavior."""

import json
from dataclasses import asdict

import h5py
import numpy as np
import torch

from src.datasets.gqa_full import GQAFullDataset
from src.datasets.gqa_pilot import record_identity
from src.datasets.schemas import SceneGraphRecord, SceneObject, SceneRelation
from src.training.gqa_full import train_run
from tests.training.test_gqa_full import fixtures
from tools.run_gqa_full_ram import RAMGQADataset, graph_digest


def tiny_cache(directory):
    records = [
        SceneGraphRecord(
            "a",
            100,
            100,
            (SceneObject(1, 1, (0, 0, 20, 20)), SceneObject(2, 2, (30, 0, 50, 20))),
            (SceneRelation(1, 2, 1), SceneRelation(1, 2, 3), SceneRelation(1, 2, 1)),
        ),
        SceneGraphRecord("b", 100, 100),
        SceneGraphRecord(
            "c",
            100,
            100,
            (SceneObject(1, 1, (0, 0, 20, 20)), SceneObject(2, 2, (30, 0, 50, 20))),
            (SceneRelation(2, 1, 2),),
        ),
    ]
    entries = []
    for i, record in enumerate(records):
        name = f"shard-{i}.h5"
        with h5py.File(directory / name, "w") as f:
            group = f.create_group(record.image_id)
            group.create_dataset("record", data=json.dumps(asdict(record)).encode())
            group.create_dataset("features", data=np.full((len(record.objects), 512), 0.125, dtype=np.float16))
            group.attrs["record_sha256"] = record_identity(record)
        entries.append({"file": name, "ids": [record.image_id]})
    return {"identity": {"splits": {"train": ["c", "b", "a"]}}, "shards": entries}


def test_parallel_ram_matches_every_tensor_and_checkpoint_resume(tmp_path):
    torch.set_num_threads(1)
    metadata = tiny_cache(tmp_path)
    lazy = GQAFullDataset(tmp_path, metadata, "train")
    ram = RAMGQADataset(tmp_path, metadata, "train", workers=2)
    try:
        expected = [lazy[i].to_dict() for i in range(len(lazy))]
        before = graph_digest([g.to_dict() for g in ram[:]])
        assert before == graph_digest(expected)
        assert [g.image_id for g in ram[:]] == ["c", "b", "a"]
        assert ram[1].x.shape == (0, 517)
        assert ram[2].gt_predicate.tolist() == [1, 3, 1]
        _, _, vectors, experiment = fixtures("cpu")
        original = train_run("pe-plpr", 42, lazy, lazy, vectors, experiment, tmp_path / "uninterrupted")
        path = tmp_path / "resume"
        assert train_run("pe-plpr", 42, lazy, lazy, vectors, experiment, path, stop_after_batches=1) is None
        resumed = train_run("pe-plpr", 42, ram, ram, vectors, experiment, path)
        assert original["history"] == resumed["history"]
        assert original["validation"] == resumed["validation"]
        a = torch.load(tmp_path / "uninterrupted/last.pt", weights_only=True)
        b = torch.load(path / "last.pt", weights_only=True)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        assert before == graph_digest([g.to_dict() for g in ram[:]])
    finally:
        lazy.close()
        ram.close()


def test_graph_digest_detects_any_feature_change(tmp_path):
    metadata = tiny_cache(tmp_path)
    lazy = GQAFullDataset(tmp_path, metadata, "train")
    try:
        item = lazy[0].to_dict()
        before = graph_digest([item])
        item["x"][0, 0] += 1
        assert graph_digest([item]) != before
    finally:
        lazy.close()
