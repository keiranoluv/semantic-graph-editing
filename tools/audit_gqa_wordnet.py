"""Recheck KG pilot evidence and preservation of the fifteen completed GQA runs."""

import argparse
import json
from pathlib import Path

import torch

from src.datasets.gqa_full import atomic_write, digest
from src.knowledge.transe.core import sha256
from tools.export_gqa_wordnet import load_features
from tools.run_gqa_kg_pilot import KINDS, assert_exact, verify_reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("runs/level2/gqa-wordnet-pilot-v1"))
    parser.add_argument("--report", type=Path, default=Path("reports/level2/gqa-wordnet-v1"))
    args = parser.parse_args(argv)
    reference_path = Path("runs/level2/gqa-full-v3-bs512/experiment.json")
    reference = json.loads(reference_path.read_text())
    verify_reference(reference)
    new_sources = sorted(str(p) for p in Path("src").rglob("*.py") if str(p) not in reference["code_sha256"])
    if new_sources:
        raise ValueError("Unexpected new src files would change historical launcher identity")
    prior_verification = json.loads(Path("reports/level2/gqa-full-v3-results-v1/verification.json").read_text())
    historical = []
    for run in prior_verification["runs"]:
        result_path = Path(run["result_path"])
        result = json.loads(result_path.read_text())
        if (
            sha256(result_path) != run["result_sha256"]
            or sha256(result_path.parent / "best.pt") != run["best_checkpoint_sha256"]
            or result["status"] != "PASS"
            or not (result_path.parent / "last.pt").is_file()
        ):
            raise ValueError(f"Historical result/checkpoint changed: {result_path}")
        historical.append(
            {
                "model": run["model"],
                "seed": run["seed"],
                "result_sha256": run["result_sha256"],
                "best_checkpoint_sha256": run["best_checkpoint_sha256"],
                "status": "PASS",
            }
        )
    pilot = json.loads((args.pilot / "experiment.json").read_text())
    verify_reference(pilot)
    feature_folder = Path("data/knowledge/gqa-wordnet-features-draft-v1")
    feature_manifest_path = feature_folder / "manifest.json"
    features, valid, _ = load_features(feature_folder, Path(reference["config"]["manifest"]))
    if sha256(feature_manifest_path) != pilot["feature_manifest_sha256"]:
        raise ValueError("Pilot feature manifest changed")
    results = {}
    for kind in KINDS:
        folder = args.pilot / kind / "42"
        result = json.loads((folder / "result.json").read_text())
        if (
            result["identity"] != {"experiment_sha256": digest(pilot), "model": kind, "seed": 42}
            or result["status"] != "PASS"
            or sha256(folder / "best.pt") != result["checkpoint_sha256"]
            or result["test_used"] is not False
        ):
            raise ValueError(f"Invalid pilot result: {kind}")
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        if kind != "gat":
            assert_exact(checkpoint["model"]["kg_features"], features)
            assert_exact(checkpoint["model"]["kg_valid"], valid)
        if checkpoint["score"] != result["validation"]["gc"]["mR@50"]:
            raise ValueError("Checkpoint selection score mismatch")
        results[kind] = result
    traces = [[row["sampling_trace_sha256"] for row in r["history"]] for r in results.values()]
    assert_exact(traces[0], traces[1])
    assert_exact(traces[0], traces[2])
    assert_exact(results["gat-kg"]["initial_state_sha256"], results["gat-kg-off"]["initial_state_sha256"])
    a, b = [
        torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        for folder in (args.pilot / "gat-kg/42", args.pilot / "resume-proof/gat-kg/42")
    ]
    for key in ("model", "optimizer", "torch_rng", "cuda_rng", "sampler_rng", "best_model", "history"):
        assert_exact(a[key], b[key])
    notebook_hash = sha256(Path("notebooks/level1_scene_graph_demo.ipynb"))
    if notebook_hash != "786c3eff8b7d4abd93c1546dba3b9d9d4ff635fc160fa80a59c0eefdd8952c19":
        raise ValueError("Personal notebook differs from handoff; inspect before recording preservation")
    proof = {
        "status": "PASS",
        "frozen_reference_source_file_count": len(reference["code_sha256"]),
        "all_reference_source_hashes_unchanged": True,
        "new_src_files_added": new_sources,
        "historical_runs": historical,
        "notebook_sha256": notebook_hash,
        "notebook_hash_matches_handoff": True,
        "pilot_experiment_sha256": digest(pilot),
        "pilot_result_and_checkpoint_hashes_match": True,
        "pilot_kg_buffers_match_export": True,
        "pilot_sampling_identical_across_controls": True,
        "pilot_kg_on_off_initial_states_identical": True,
        "pilot_exact_resume_state": True,
        "embedding_training_source_sha256": {
            name: sha256(Path(name)) for name in ("scripts/train_transe.py", "src/knowledge/transe/core.py")
        },
        "embedding_checkpoint_sha256": sha256(Path("runs/level2/gqa-wordnet-transe-draft-v1/best.pt")),
        "feature_manifest_sha256": sha256(feature_manifest_path),
        "verification_source_sha256": sha256(Path(__file__)),
        "gqa_test_used": False,
        "kg_only_heldout_triples_evaluated": True,
    }
    atomic_write(args.report / "verification.json", proof)
    print(
        "PASS: 15 historical runs, 73 frozen sources, personal notebook, KG alignment, sampling and exact pilot resume"
    )


if __name__ == "__main__":
    main()
