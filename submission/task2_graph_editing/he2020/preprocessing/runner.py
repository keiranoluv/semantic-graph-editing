"""CPU diagnostic and explicit human-gated expansion; never trains a model."""

import argparse
import resource
import time
from pathlib import Path

from task2_graph_editing.he2020.preprocessing import VERSION
from task2_graph_editing.he2020.preprocessing.compiler import compile_record
from task2_graph_editing.he2020.preprocessing.data import leakage_filter, load_examples
from task2_graph_editing.he2020.preprocessing.dataset import (
    ALIGNMENT,
    ARCHIVE,
    ARCHIVE_SHA256,
    OUTPUT,
    POLICY,
    code_identity,
    digest,
    download,
    encode,
    fixed,
    inspect,
    prepare,
    prepare_expansion,
    read,
    sha256,
    write,
)
from task2_graph_editing.he2020.preprocessing.report import summarize
from task2_graph_editing.he2020.preprocessing.review import audit, build, make_packet, serve


def compile_dataset(output):
    output = Path(output)
    started = time.monotonic()
    receipt = read(output / "raw-receipt.json")
    if (
        receipt["archive_sha256"] != ARCHIVE_SHA256
        or receipt["cohort_sha256"] != sha256(output / "cohort.json")
        or receipt["raw_sha256"] != sha256(output / "raw-records.json")
    ):
        raise ValueError("raw_receipt_mismatch")
    if (output / "summary.json").exists():
        manifest = read(output / "manifest.json")
        if manifest["code"]["files"] != code_identity()["files"]:
            raise ValueError("adapter_code_changed_use_new_versioned_output_keep_cohort")
        if not (output / "output-receipt.json").exists():
            summary = read(output / "summary.json")
            if summary["manifest_sha256"] != sha256(output / "manifest.json") or summary["edits_sha256"] != sha256(
                output / "edits.json"
            ):
                raise ValueError("incomplete_compilation_evidence_changed")
            load_examples(output)
            write(
                output / "output-receipt.json",
                {
                    "files": {
                        name: sha256(output / name)
                        for name in ("manifest.json", "edits.json", "policy-B.json", "summary.json")
                    }
                },
            )
        for name, checksum in read(output / "output-receipt.json")["files"].items():
            if sha256(output / name) != checksum:
                raise ValueError("compilation_output_hash_changed")
        load_examples(output)
        return read(output / "summary.json")
    raw = read(output / "raw-records.json")
    cohort = read(output / "cohort.json")
    if len(raw) != cohort["size"] or len({r["id"] for r in raw}) != len(raw):
        raise ValueError("cohort_size_or_ids_changed")
    if "indices" in cohort:
        if (
            len(raw) != 100
            or cohort["seed"] != 42
            or [r["original_record_index"] for r in raw] != cohort["indices"]
            or any(r["official_split"] != "train" for r in raw)
        ):
            raise ValueError("pilot_requires_fixed100_real_train_seed42")
    records = [compile_record(r) for r in raw]
    alternative = [compile_record(r, "B") for r in raw]
    excluded = leakage_filter(records)
    for row in records:
        if row["id"] in excluded:
            row["problems"].append(excluded[row["id"]])
            row.update(status="rejected", conversion_status="rejected", eligible_for_training=False)
            row["stages"]["candidate_acceptance"] = False
    keys = (
        "id",
        "source_dataset",
        "source_variant",
        "split",
        "official_split",
        "original_record_index",
        "instruction",
        "source_group_id",
        "source_instruction_sha256",
    )
    manifest = {
        "schema": VERSION,
        "archive_sha256": ARCHIVE_SHA256,
        "policy": POLICY,
        "alignment": ALIGNMENT,
        "seed": 42,
        "code": code_identity(),
        "inputs": {
            name: sha256(output / name)
            for name in ("raw-receipt.json", "raw-records.json", "cohort.json", "inspection.json")
        },
        "records": [
            {**{k: row.get(k) for k in keys}, "raw_evidence_sha256": digest(original)}
            for row, original in zip(records, raw, strict=True)
        ],
        "splits": {s: [r["id"] for r in records if r["split"] == s] for s in ("train", "val", "test")},
        "review_ids": sorted(
            (r["id"] for r in raw if r["split"] == "train"), key=lambda i: digest(["diagnostic-review", 42, i])
        )[:20],
        "official_test_content_opened": False,
        "assignment_source_approval": "pending_instructor",
    }
    write(output / "manifest.json", manifest)
    edits = {"schema": VERSION, "dataset_manifest_sha256": sha256(output / "manifest.json"), "records": records}
    write(output / "edits.json", edits)
    write(output / "policy-B.json", {"records": alternative})
    _, loader_report = load_examples(output)
    summary = summarize(records, alternative)
    summary.update(
        adapter_version=VERSION,
        archive_sha256=ARCHIVE_SHA256,
        cohort_sha256=sha256(output / "cohort.json"),
        manifest_sha256=sha256(output / "manifest.json"),
        edits_sha256=sha256(output / "edits.json"),
        code=manifest["code"],
        loader=loader_report,
        runtime_compile_seconds=time.monotonic() - started,
        runtime_prepare_seconds=receipt.get("runtime_seconds"),
        peak_parent_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_reader_process_rss_kib=max(s["peak_worker_rss_kib"] for s in receipt["scans"].values()),
        runtime_scope="prepare includes bounded archive inspection and scans; compile includes both policies and loader; download/test/review time excluded",
        duplicate_exclusions=excluded,
        official_test_content_opened=False,
    )
    write(output / "summary.json", summary)
    write(
        output / "output-receipt.json",
        {
            "files": {
                name: sha256(output / name) for name in ("manifest.json", "edits.json", "policy-B.json", "summary.json")
            }
        },
    )
    return summary


def expansion(output, archive, destination, directory, csv_path, train_size=500, dev_size=100):
    if csv_path is None:
        raise ValueError("expansion_requires_real_human_review_csv")
    gate = audit(output, directory, csv_path)
    if not gate["ready"]:
        raise ValueError("pilot_human_review_gate_not_ready")
    if Path(output).resolve() == Path(destination).resolve():
        raise ValueError("expansion_requires_separate_output")
    evidence = {
        "manifest_sha256": sha256(Path(output) / "manifest.json"),
        "review_packet_sha256": gate["review_packet_sha256"],
        "review_csv_sha256": gate["review_csv_sha256"],
        "audit": gate,
    }
    prepare_expansion(archive, destination, train_size, dev_size, evidence)
    return compile_dataset(destination)


def export(output, destination, review_output, directory, csv_path):
    if csv_path is None:
        raise ValueError("export_requires_real_human_review_csv")
    gate = audit(review_output, directory, csv_path)
    if not gate["ready"]:
        raise ValueError("human_review_gate_not_ready")
    current, pilot = make_packet(output), make_packet(review_output)
    # Transfer review only when every selected label has byte-identical evidence.
    from task2_graph_editing.he2020.preprocessing.data import checked_inputs

    _, _, labels = checked_inputs(output)
    for row in pilot["cases"]:
        if row["id"] not in labels or digest(labels[row["id"]]) != row["evidence_sha256"]:
            raise ValueError("pilot_review_evidence_not_preserved_in_expansion")
    examples, report = load_examples(output, rejected=gate["rejected_ids"])
    if sum(map(len, examples.values())) < 300 or any(not v for v in examples.values()):
        raise ValueError("export_requires300_usable_train_val_with_official_separation")
    payload = {
        "schema": "emnlp2020_task_b_examples_v1",
        "manifest_sha256": current["dataset_manifest_sha256"],
        "edits_sha256": current["edits_sha256"],
        "review": gate,
        "loader_report": report,
        "assignment_dataset_source_requirement_satisfied": False,
        "instructor_approval": "pending",
        "records": [
            {
                "id": e.id,
                "split": e.split,
                "source_group_id": e.group,
                "instruction": e.instruction,
                "source_graph": e.source.to_dict(),
                "target_graph": e.target.to_dict(),
                "operations": [o.to_dict() for o in e.operations],
            }
            for rows in examples.values()
            for e in rows
        ],
    }
    write(destination, payload)
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "examples": report["examples"],
        "instructor_approval": "pending",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "inspect",
            "plan",
            "prepare",
            "pilot",
            "compile",
            "review-build",
            "review-serve",
            "review-audit",
            "expand",
            "export",
        ),
    )
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--reviews-csv", type=Path)
    parser.add_argument("--review-output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--dev-size", type=int, default=100)
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args(argv)
    directory = args.review_dir or (Path("reviews") / args.output.name)
    try:
        if args.download:
            download(args.archive)
        if args.stage == "inspect":
            result = inspect(args.archive)
            write(args.output / "inspection.json", result)
        elif args.stage == "prepare":
            result = prepare(args.archive, args.output)
        elif args.stage in ("pilot", "compile"):
            if args.stage == "pilot":
                prepare(args.archive, args.output)
            result = compile_dataset(args.output)
            if args.report_dir:
                for name in ("inspection.json", "cohort.json", "summary.json", "output-receipt.json"):
                    fixed(args.report_dir / name, (args.output / name).read_bytes())
        elif args.stage == "review-build":
            result = build(args.output, directory)
        elif args.stage == "review-serve":
            serve(args.output, directory, args.port)
            return 0
        elif args.stage == "review-audit":
            result = audit(args.output, directory, args.reviews_csv)
        elif args.stage == "expand":
            if args.destination is None:
                raise ValueError("explicit_separate_destination_required")
            result = expansion(
                args.output, args.archive, args.destination, directory, args.reviews_csv, args.train_size, args.dev_size
            )
        elif args.stage == "export":
            if args.destination is None:
                raise ValueError("explicit_export_destination_required")
            result = export(
                args.output, args.destination, args.review_output or args.output, directory, args.reviews_csv
            )
        else:
            reasons = []
            if not (args.output / "summary.json").exists():
                reasons.append("real_pilot_missing")
            if args.reviews_csv is None:
                reasons.append("human_reviews_missing")
            else:
                gate = audit(args.output, directory, args.reviews_csv)
                if not gate["ready"]:
                    reasons.append("human_review_gate_not_ready")
            result = {
                "ready_for_expansion": not reasons,
                "blockers": reasons,
                "instructor_approval": "pending",
                "gpu_tasks": 0,
                "automatic_queue": False,
            }
        # Large diagnostics live in immutable files, not terminal output.
        brief = {k: v for k, v in result.items() if k not in ("code", "archive_members", "scans", "members", "records")}
        print(encode(brief).decode(), end="")
        return 2 if args.stage == "plan" and not result["ready_for_expansion"] else 0
    except (ValueError, OSError) as error:
        print(encode({"error": str(error), "stage": args.stage}).decode(), end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
