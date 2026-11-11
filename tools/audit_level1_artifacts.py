"""Read-only Level 1 artifact audit; run as python -m tools.audit_level1_artifacts.

Downloads pinned reference source and writes reports/level1/audit-2026-11-11.json.
Does not evaluate full datasets or modify models, checkpoints, or prior results.
"""
import ast
import json
import tempfile
from collections import Counter
from functools import reduce
from pathlib import Path
from urllib.request import urlopen

import h5py
import numpy as np
import torch

from scripts.prepare_pair_kg_prior import make_prior
from src.evaluation.scene_graph import summarize_evaluation
from src.features.visual import load_visual_manifest
from src.knowledge.export import load_class_features
from src.knowledge.transe.core import sha256


def main():
    REFERENCE_COMMIT = "ceb71fa88461c2a97a6258a80f47669d89207296"
    reference_dir = Path(tempfile.mkdtemp(prefix="level1-reference-"))
    for filename, remote in [("sgg_eval.py", "evaluation/vg/sgg_eval.py"), ("visual_genome.py", "visual_genome.py")]:
        url = f"https://raw.githubusercontent.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/{REFERENCE_COMMIT}/maskrcnn_benchmark/data/datasets/{remote}"
        with urlopen(url, timeout=30) as response:
            (reference_dir / filename).write_bytes(response.read())
    ROOT = Path.cwd()
    D = ROOT / "data/raw/vg150"
    out = {}
    # Run the actual reference mean-recall class in isolation (no detector dependencies).
    src = (reference_dir / "sgg_eval.py").read_text()
    tree = ast.parse(src)

    class Base:
        def __init__(self, result_dict):
            self.result_dict = result_dict

    scope = {"SceneGraphEvaluation": Base, "np": np, "reduce": reduce}
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "SGMeanRecall")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "reference SGMeanRecall", "exec"), scope)

    def probe(num_rel):
        ref = scope["SGMeanRecall"]({}, num_rel, ["background"] + [str(i) for i in range(1, num_rel)])
        ref.register_container("predcls")
        rows = []
        for count, hit in [(9, 9), (1, 0)]:
            gt = np.array([[i, i + 1, 1] for i in range(count)])
            ref.collect_mean_recall_items({}, {"gt_rels": gt, "pred_to_gt": [list(range(hit))]}, "predcls")
            rows.append(
                {
                    "support": count,
                    "hits": {50: hit},
                    "class_support": Counter({1: count}),
                    "class_hits": {50: Counter({1: hit})},
                }
            )
        ref.calculate_mean_recall("predcls")
        return {
            "project": summarize_evaluation(rows, ks=(50,))["mR@50"],
            "reference": ref.result_dict["predcls_mean_recall"][50],
        }

    out["reference"] = {
        "repository": "KaihuaTang/Scene-Graph-Benchmark.pytorch",
        "commit": REFERENCE_COMMIT,
        "sgg_eval_sha256": sha256((reference_dir / "sgg_eval.py")),
        "visual_genome_sha256": sha256((reference_dir / "visual_genome.py")),
    }
    out["mean_recall_probe"] = {"one_supported_class": probe(2), "one_supported_one_absent": probe(3)}
    s = json.loads((D / "splits.json").read_text())
    meta = json.loads((D / "image_data.json").read_text())
    with h5py.File(D / "VG-SGG-with-attri.h5") as h:
        flags = h["split"][:]
        eligible = (h["img_to_first_box"][:] >= 0) & (h["img_to_first_rel"][:] >= 0)
        train = np.flatnonzero((flags == 0) & eligible).tolist()
        test = np.flatnonzero((flags == 2) & eligible).tolist()
        expected = {"train": train[5000:], "val": train[:5000], "test": test}
        out["split"] = {
            "counts": {k: len(v) for k, v in s.items()},
            "matches_reference_before_optional_nonoverlap_filter": s == expected,
            "unique_image_ids": len({meta[i]["image_id"] for v in s.values() for i in v}) == sum(map(len, s.values())),
        }
        # Count distinct multi-label supervision loss directly from train annotations.
        rel = h["relationships"][:]
        pred = h["predicates"][:].reshape(-1)
        starts = h["img_to_first_rel"][:]
        ends = h["img_to_last_rel"][:]
        multi = total_pairs = annotations = lost = 0
        for row in s["train"]:
            by = {}
            for pair, p in zip(rel[starts[row] : ends[row] + 1], pred[starts[row] : ends[row] + 1], strict=False):
                by.setdefault(tuple(pair), []).append(int(p))
            total_pairs += len(by)
            annotations += sum(map(len, by.values()))
            for v in by.values():
                if len(set(v)) > 1:
                    multi += 1
                    lost += len(v)
        out["ambiguous_training_pairs"] = {
            "total_positive_pairs": total_pairs,
            "multi_predicate_pairs": multi,
            "annotations": annotations,
            "annotations_on_ignored_pairs": lost,
        }
    plan = json.loads(Path("reports/level1/clip-all-27run-test-plan.json").read_text())
    paths = (
        [Path(e["validation_report"]) for e in plan["entries"]]
        + list(Path("reports/level1/test-v1").glob("*.json"))
        + list(Path("reports/level1").glob("clip-pair-kg-v1-*-val5000.json"))
        + list(Path("reports/level1").glob("clip-fusion-v1-*-seed*-val5000.json"))
    )
    # Pair report naming differs; discover by model and individual checkpoint field.
    for p in Path("reports/level1").glob("*pair*val5000.json"):
        r = json.loads(p.read_text())
        if "checkpoint" in r:
            paths.append(p)
    paths = sorted(set(paths))
    failures = []
    groups = Counter()
    common = {}
    cache = {}
    for p in paths:
        r = json.loads(p.read_text())
        cp = Path(r["checkpoint"])
        c = cache.get(str(cp))
        if c is None:
            c = torch.load(cp, map_location="cpu", weights_only=True)
            cache[str(cp)] = c
        checks = {
            "sha256": sha256(cp) == r["checkpoint_sha256"],
            "epoch": c["epoch"] == r["checkpoint_epoch"],
            "config": c["config"] == r["config"],
            "data_hash": r["data_sha256"] == plan["data_sha256"],
            "full_split": r["full_split"] and r["images_evaluated"] == len(s[r["split"]]),
            "clip_hash": c["config"]["visual_cache_manifest_sha256"] == plan["visual_cache_manifest_sha256"],
        }
        if r["split"] == "val":
            checks["checkpoint_metrics"] = all(
                abs(c["validation"][m] - r[m]) < 1e-10 for m in ["R@50", "R@100", "mR@50", "mR@100"]
            )
        for k, v in checks.items():
            if not v:
                failures.append({"report": str(p), "check": k})
        groups[r["model"] + ":" + r["split"]] += 1
        for key in [
            "train_limit",
            "val_limit",
            "negative_ratio",
            "batch_size",
            "lr",
            "weight_decay",
            "geometry_dim",
            "box_policy",
            "selection_metric",
        ]:
            common.setdefault(key, set()).add(str(c["config"].get(key)))
    out["report_integrity"] = {
        "count": len(paths),
        "groups": dict(groups),
        "failures": failures,
        "shared_config_values": {k: sorted(v) for k, v in common.items()},
    }
    out["test_plan_checkpoint_hashes_match"] = all(
        sha256(Path(e["checkpoint"])) == e["checkpoint_sha256"] for e in plan["entries"]
    )
    for filename, expected_hash in plan["data_sha256"].items():
        assert sha256(D / filename) == expected_hash
    prior_path = Path("reports/level1/clip-pair-kg-train-prior-v1.json")
    out["train_prior_recomputed_equal"] = make_prior(D, len(s["train"]), 3) == json.loads(prior_path.read_text())
    f, v, m = load_class_features(Path("data/knowledge/c2-conceptnet-toan-v3"))
    c = torch.load("runs/transe/conceptnet-toan-v3-seed42/best.pt", map_location="cpu", weights_only=True)
    r = c["model"]["relation.weight"]
    scores = torch.load(
        "data/knowledge/c2-conceptnet-relation-scores-toan-v1/scores.pt", map_location="cpu", weights_only=True
    )["scores"]
    mask = v[:, None] & v[None, :]
    diffs = {}
    for norm in [1, 2]:
        a = []
        for rel in r:
            raw = -(f[:, None, :] + rel - f[None, :, :]).norm(p=norm, dim=-1)
            a.append(torch.where(mask, (raw - raw[mask].mean()) / raw[mask].std(unbiased=False).clamp_min(1e-6), 0))
        diffs[str(norm)] = float((torch.stack(a, dim=-1) - scores).abs().max())
    out["relation_score_max_difference"] = diffs

    load_visual_manifest(ROOT / "data/cache/vg150-clip-vit-b32-v1", D, "clip")
    out["visual_cache_content_hash_verified"] = True
    bundles = {sha256(p): p.parent for p in Path("data/knowledge").glob("*/manifest.json")}
    checked_bundles = {}
    buffer_checks = []
    for _cp, c in cache.items():
        if c["config"]["model"] != "gat_kg":
            continue
        digest = c["config"]["kg_manifest_sha256"]
        if digest not in checked_bundles:
            checked_bundles[digest] = load_class_features(bundles[digest])
        features, valid, manifest = checked_bundles[digest]
        buffer_checks.append(
            torch.equal(features, c["model"]["kg_features"]) and torch.equal(valid, c["model"]["kg_valid"])
        )
    out["kg_buffers"] = {
        "checkpoints_checked": len(buffer_checks),
        "all_match_c2": all(buffer_checks),
        "bundles": [
            {"path": str(bundles[d]), "valid_class_count": m["valid_class_count"]}
            for d, (_, _, m) in checked_bundles.items()
        ],
    }
    out["scope"] = "CPU artifact integrity and synthetic protocol probes; no full-split inference or retraining"
    Path("reports/level1/audit-2026-11-11.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
