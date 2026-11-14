"""Reproduce lexical pilot comparisons from sealed predictions on CPU only."""
from collections import Counter
from pathlib import Path

from tools.edit_model_v1.codec import canonical_actions, decode_actions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256, write
from tools.magicbrush_compact_pilot_v1.data import evaluation_report
from tools.magicbrush_compact_pilot_v1.runner import verify_completed as verify_baseline
from tools.magicbrush_experiment_v1.evaluation import score_rows
from tools.magicbrush_lexical_pilot_v1.analysis import diagnose
from tools.magicbrush_lexical_pilot_v1.features import collate_inputs
from tools.magicbrush_lexical_pilot_v1.runner import completed, preflight
from tools.prepare_magicbrush import ROOT, digest


def main():
    cfg, examples, validation, store, identity, _ = preflight()
    output = ROOT / cfg["output"]
    summary = read(output / "summary.json")
    assert read(output / "experiment.json") == identity
    assert summary["experiment_sha256"] == digest(identity)
    base_root = (ROOT / cfg["provenance"]["baseline_summary"]["path"]).parent
    baseline = read(base_root / "summary.json")
    base_identity = {**read(base_root / "experiment.json"), "frozen_features": read(base_root / "feature-receipt.json")}
    assert digest(base_identity) == baseline["training_identity_sha256"]
    vocabulary = identity["vocabulary"]["labels"]
    feature_counts = Counter()
    for e in examples["val"]:
        features = collate_inputs([e.model_input()], store)
        for op in e.operations:
            if op.kind != "add_node":
                continue
            feature_counts["reference_added_nodes"] += 1
            if op.after.label not in vocabulary:
                feature_counts["added_names_outside_train_vocabulary"] += 1
            else:
                feature_counts["added_names_in_train_vocabulary"] += 1
                feature_counts["in_vocabulary_names_exactly_mentioned"] += bool(
                    features["label_grounding"][0, vocabulary.index(op.after.label), 0]
                )
    audit = {
        "scope": "Sealed prediction audit; no new GPU/model inference and no test labels",
        "experiment_sha256": digest(identity),
        "execution": {
            "training_code_commit": "cb1ef02",
            "gpu": "NVIDIA GeForce RTX 5090",
            "user_authorized_assistant_gpu": True,
            "gpu_training_runs": 2,
            "sequential_models": ["mlp", "gnn"],
            "seed": 42,
            "completed": True,
        },
        "checks": [
            "current/parent releases and frozen cache verified by preflight",
            "both baseline and lexical completion receipts verified",
            "main raw/constrained predictions rescored with identical 64-case labels",
            "all 67 prediction IDs/schema-gap accounting verified",
            "grounding diagnostics recomputed",
            "ablation artifacts hash-verified; metrics recorded during committed inference runs",
        ],
        "feature_availability_on_validation": dict(feature_counts),
        "sources_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [output / "summary.json", base_root / "summary.json", Path(__file__).resolve()]
        },
        "models": {},
        "interpretation_vi": "Tín hiệu từ vựng giúp chọn đúng tên thêm mới hơn, nhưng ít chuyển thành thao tác đúng đầy đủ. Mức tăng F1 có giới hạn; chỉ một seed, không kết luận ưu thế thống kê hoặc GNN tốt hơn MLP.",
        "remaining_limitations": [
            "closed train label vocabulary: gorilla/guitar cannot be emitted",
            "word mentions do not identify semantic role: road can be chosen instead of requested gorilla",
            "exact added-node attributes and relations remain difficult",
            "three validation schema gaps remain unchanged",
            "inference ablation is conditional on trained weights, not independently retrained models",
        ],
        "next_step_vi": "Giữ nhãn và kết quả hiện tại làm đối chứng. Nếu tiếp tục, ưu tiên cơ chế sao chép tên ngoài từ vựng từ chính câu yêu cầu và phân biệt đối tượng cần thêm với đối tượng làm mốc quan hệ; không dùng target để tạo ứng viên. Chưa mở rộng seed/test hoặc tự giảm thêm nhãn.",
    }
    for kind in cfg["models"]:
        directory = output / kind / "42"
        result, measured, ablation = completed(
            directory, {"experiment_sha256": digest(identity), "model": kind, "seed": 42}
        )
        b_result, b_measured = verify_baseline(
            base_root / kind / "42", {"experiment_sha256": digest(base_identity), "model": kind, "seed": 42}
        )
        rows = read(directory / "predictions-val-all.json")
        old_rows = read(base_root / kind / "42/predictions-val-all.json")
        recomputed = evaluation_report(validation, rows)
        assert all(measured[k] == v for k, v in recomputed.items())
        assert evaluation_report(validation, old_rows) == b_measured
        assert measured == summary["models"][kind]["evaluation"]
        assert ablation == summary["models"][kind]["ablation"]
        lookup, old_lookup = ({r["id"]: r for r in rs} for rs in (rows, old_rows))
        current_diag = diagnose(examples["val"], [lookup[e.id] for e in examples["val"]])
        old_diag = diagnose(examples["val"], [old_lookup[e.id] for e in examples["val"]])
        assert digest(current_diag) == digest(measured["grounding_diagnostics"])
        exact, laptop, attributes = [], [], Counter()
        for e in examples["val"]:
            row = lookup[e.id]
            if score_rows([e], [row])["exact_target_graph_accuracy"] == 1:
                exact.append(
                    {"id": e.id, "instruction": e.instruction, "operations": [o.to_dict() for o in e.operations]}
                )
            pred = canonical_actions(e.source, decode_actions(row["tokens"], e.source))
            expected = Counter((o.after.label, o.after.attributes) for o in e.operations if o.kind == "add_node")
            actual = Counter((o.after.label, o.after.attributes) for o in pred if o.kind == "add_node")
            attributes["added_name_and_attributes_match_ignoring_id"] += sum((expected & actual).values())
            if "laptop on the table" in e.instruction:
                laptop.append(
                    {
                        "id": e.id,
                        "instruction": e.instruction,
                        "expected": [o.to_dict() for o in e.operations],
                        "predicted": [o.to_dict() for o in pred],
                    }
                )
        before = b_measured["constrained"]["micro"]["f1"]
        after = measured["constrained"]["micro"]["f1"]
        audit["models"][kind] = {
            "selected_epoch": result["selected_epoch"],
            "epochs_completed": result["epochs_completed"],
            "stopping_reason": result["stopping_reason"],
            "baseline_f1": before,
            "lexical_f1": after,
            "delta_percentage_points": 100 * (after - before),
            "baseline": b_measured,
            "lexical": measured,
            "baseline_diagnostics": old_diag,
            "lexical_diagnostics": current_diag,
            "added_attribute_diagnostics": dict(attributes),
            "exact_cases": exact,
            "laptop_case": laptop,
            "inference_ablations": {
                name: {"f1": value["constrained"]["micro"]["f1"], "diagnostics": value["diagnostics"]}
                for name, value in ablation["results"].items()
            },
            "completion_receipt_sha256": sha256(directory / "completed-receipt.json"),
            "train_val_loss_selected_and_last": [
                {k: r[k] for k in ("epoch", "train_structured_loss", "validation_structured_loss")}
                for r in result["history"]
                if r["epoch"] in (result["selected_epoch"], result["epochs_completed"])
            ],
        }
    target = ROOT / "reports/level2/magicbrush-lexical-pilot-v1/result-audit.json"
    write(target, audit)
    print("Saved verified audit:", target)
    print("Validation feature coverage:", dict(feature_counts))
    for kind, result in audit["models"].items():
        print(
            kind,
            "F1",
            result["baseline_f1"],
            "->",
            result["lexical_f1"],
            "exact cases",
            result["exact_cases"],
            "laptop",
            result["laptop_case"],
        )


if __name__ == "__main__":
    main()
