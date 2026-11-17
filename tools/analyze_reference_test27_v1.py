"""CPU-only audit and analysis of completed reference test reports."""
import csv
import io
import json
import statistics
from pathlib import Path

from tools.level1_reference_test27_v1 import METRICS, OUTPUT, PLAN, PLAN_HASH, ROOT, sha256, validate


def main():
    assert sha256(PLAN) == PLAN_HASH
    plan = json.loads(PLAN.read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["plan_sha256"] == PLAN_HASH
    indexed = {r["path"]: r["sha256"] for r in summary["reports"]}
    groups, rows = {}, []
    for entry in plan["entries"]:
        path = OUTPUT / f"{entry['condition']}-seed{entry['seed']}-test26446.json"
        row = json.loads(path.read_text())
        validate(row, entry, plan)
        assert sha256(ROOT / entry["checkpoint"]) == entry["checkpoint_sha256"]
        assert indexed[str(path.relative_to(ROOT))] == sha256(path)
        assert row["num_images"] == row["num_images_with_gt"] == 26446
        assert row["num_predicates_with_gt"] == 50
        groups.setdefault(entry["condition"], []).append(row)
        rows.append({"condition": entry["condition"], "seed": entry["seed"], **{k: row[k] * 100 for k in METRICS}})
    assert len(rows) == len(indexed) == 27
    for group in summary["groups"]:
        values = groups[group["condition"]]
        assert sorted(r["seed"] for r in values) == [42, 123, 3407]
        for key in METRICS:
            assert abs(statistics.mean(r[key] for r in values) - group["metrics"][key]["mean"]) < 1e-12
            assert abs(statistics.stdev(r[key] for r in values) - group["metrics"][key]["sample_std"]) < 1e-12
    comparisons = []
    pairs = [
        ("gat", "mlp_semantic"),
        ("wordnet_kg_on", "kg_off"),
        ("wordnet_kg_on", "wordnet_edges_off"),
        ("conceptnet_transe_kg_on", "conceptnet_transe_edges_off"),
    ]
    pairs += [(c, "gat") for c in groups if c.endswith("kg_on")]
    for treatment, control in pairs:
        a = {r["seed"]: r for r in groups[treatment]}
        b = {r["seed"]: r for r in groups[control]}
        delta = {k: [(a[s][k] - b[s][k]) * 100 for s in (42, 123, 3407)] for k in METRICS}
        comparisons.append(
            {
                "treatment": treatment,
                "control": control,
                "paired_delta_pp": delta,
                "mean_delta_pp": {k: statistics.mean(v) for k, v in delta.items()},
            }
        )
    val_delta = []
    for condition, values in groups.items():
        for row in values:
            val = json.loads(
                (ROOT / f"reports/level1/reference-v1/{condition}-seed{row['seed']}-val5000.json").read_text()
            )
            assert val["checkpoint_sha256"] == row["checkpoint_sha256"]
            assert val["protocol"] == row["protocol"]
            val_delta.append(
                {"condition": condition, "seed": row["seed"], **{k: 100 * (row[k] - val[k]) for k in METRICS}}
            )
    audit = {
        "status": "PASS",
        "reports_checked": 27,
        "images_per_report": 26446,
        "checkpoint_hashes_checked": 27,
        "summary_recomputed": True,
        "plan_sha256": PLAN_HASH,
        "summary_sha256": sha256(OUTPUT / "summary.json"),
        "analysis_source_sha256": sha256(Path(__file__)),
        "gpu_inference_performed": False,
        "seeds": [42, 123, 3407],
    }
    outputs = {
        "audit.json": json.dumps(audit, indent=2) + "\n",
        "comparisons.json": json.dumps(
            {"unit": "percentage_points", "comparisons": comparisons, "test_minus_validation": val_delta}, indent=2
        )
        + "\n",
    }
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["condition", "seed", *METRICS], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    outputs["per-seed.csv"] = buf.getvalue()
    for name, data in outputs.items():
        path = OUTPUT / name
        if path.exists():
            assert path.read_text() == data, f"Refusing to overwrite changed artifact: {path}"
        else:
            path.write_text(data)
    print("| Model | R@50 | R@100 | mR@50 | mR@100 |")
    print("|---|---:|---:|---:|---:|")
    for group in summary["groups"]:
        print(
            "| "
            + group["condition"]
            + " | "
            + " | ".join(
                f"{group['metrics'][k]['mean']*100:.2f} ± {group['metrics'][k]['sample_std']*100:.2f}" for k in METRICS
            )
            + " |"
        )
    for item in comparisons:
        print(
            item["treatment"],
            "-",
            item["control"],
            item["mean_delta_pp"],
            "paired mR50",
            item["paired_delta_pp"]["mR@50"],
        )
    print(
        "TEST MINUS VAL mR50:",
        {c: statistics.mean(r["mR@50"] for r in val_delta if r["condition"] == c) for c in groups},
    )


if __name__ == "__main__":
    main()
