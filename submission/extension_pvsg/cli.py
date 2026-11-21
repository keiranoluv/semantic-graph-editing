"""CPU-only M1 demo, evaluation and train-annotation structural audit."""

import argparse
import json
from pathlib import Path

from extension_pvsg.contracts import ROOT, read_json
from extension_pvsg.data import AnnotationIndex, split_ids
from extension_pvsg.decoding import decode_windows
from extension_pvsg.evaluation import evaluate
from extension_pvsg.graphs import batch_graphs, build_graph
from extension_pvsg.synthetic import perfect_fixture_windows, synthetic_video_targets
from extension_pvsg.validate_m0 import file_sha256


def _documents(path):
    value = read_json(path)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError("Expected a video document or list of video documents")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Artificial perfect-prediction fixture; no model")
    demo.add_argument("--output", type=Path)
    evaluation = commands.add_parser("evaluate", help="Evaluate normalized JSON documents on CPU")
    for name in ("videos", "targets", "predictions"):
        evaluation.add_argument(f"--{name}", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path)
    evaluation.add_argument("--split", choices=("train", "test", "fit", "validation"))
    evaluation.add_argument("--output", type=Path)
    audit = commands.add_parser("audit-annotation", help="Check pinned train annotation only; no media/label repair")
    audit.add_argument("--annotation", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    exit_code = 0
    if args.command == "demo":
        video, targets = synthetic_video_targets()
        predictions = decode_windows(video, perfect_fixture_windows(video, targets))
        result = evaluate([video], [targets], [predictions], expected_video_ids=[video["video_id"]])
        graph = batch_graphs([build_graph(video, start=0, stop=6), build_graph(video, start=4, stop=10)])
        result["demo"] = {
            "synthetic_only": True,
            "model_run": False,
            "nodes_in_two_windows": len(graph["nodes"]),
            "query_edges": graph["query_edges"].shape[1],
        }
    elif args.command == "evaluate":
        if bool(args.manifest) != bool(args.split):
            parser.error("--manifest and --split must be provided together")
        manifest = read_json(args.manifest) if args.manifest else None
        expected = split_ids(manifest, args.split) if manifest else None
        result = evaluate(
            _documents(args.videos), _documents(args.targets), _documents(args.predictions), expected_video_ids=expected
        )
        result["dataset"] = {
            "split_id": manifest["split_id"] if manifest else None,
            "split": args.split,
            "scope": "manifest_checked" if manifest else "custom_cohort",
        }
        result["input_sha256"] = {
            name: file_sha256(getattr(args, name)) for name in ("videos", "targets", "predictions")
        }
        if args.manifest:
            result["input_sha256"]["manifest"] = file_sha256(args.manifest)
    else:
        result = AnnotationIndex(args.annotation).audit_train_structure()
        exit_code = 2 if result["errors"] else 0
    result["provenance"] = {
        "protocol_sha256": file_sha256(ROOT / "protocol.json"),
        "vocabulary_sha256": file_sha256(ROOT / "provenance" / "vocabulary.json"),
        "implementation_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "cli.py",
                "contracts.py",
                "data.py",
                "decoding.py",
                "evaluation.py",
                "graphs.py",
                "intervals.py",
                "synthetic.py",
                "validate_m0.py",
            )
        },
        "device": "CPU",
        "training": False,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({"output": str(args.output), "exit_code": exit_code}))
    else:
        print(text, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
