"""Audit completed loss/fusion test outputs and export CPU-only comparison artifacts."""
import json
import statistics

from tools.level1_loss_fusion_test27_v1 import OUTPUT, build_plan
from tools.level1_reference_test27_v1 import METRICS, ROOT, sha256, validate


def main():
    plan = build_plan()
    frozen = OUTPUT / "frozen-checkpoints.json"
    assert json.loads(frozen.read_text()) == plan
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert summary["plan_sha256"] == sha256(frozen)
    sources = {r["path"]: r["sha256"] for r in summary["reports"]}
    groups = {}
    for e in plan["entries"]:
        path = OUTPUT / f"{e['condition']}-seed{e['seed']}-test26446.json"
        r = json.loads(path.read_text())
        validate(r, e, plan)
        assert sha256(ROOT / e["checkpoint"]) == e["checkpoint_sha256"]
        assert sha256(path) == sources[str(path.relative_to(ROOT))]
        assert r["num_images_with_gt"] == 26446 and r["num_predicates_with_gt"] == 50
        groups.setdefault(e["condition"], {})[e["seed"]] = r
    assert len(sources) == 27 and len(groups) == 9
    table = ["| Cấu hình | R@50 | R@100 | mR@50 | mR@100 |", "|---|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        rows = groups[g["condition"]]
        assert sorted(rows) == [42, 123, 3407]
        cells = []
        for k in METRICS:
            mean = statistics.mean(r[k] for r in rows.values())
            std = statistics.stdev(r[k] for r in rows.values())
            assert abs(mean - g["metrics"][k]["mean"]) < 1e-12
            assert abs(std - g["metrics"][k]["sample_std"]) < 1e-12
            cells.append(f"{mean*100:.2f} ± {std*100:.2f}")
        table.append("| " + g["condition"] + " | " + " | ".join(cells) + " |")
    pairs = [
        ("loss_adjusted_off", "loss_ce_off"),
        ("loss_adjusted_on", "loss_ce_on"),
        ("loss_ce_on", "loss_ce_off"),
        ("loss_adjusted_on", "loss_adjusted_off"),
    ]
    pairs += [(f"fusion_{m}_real", f"fusion_{c}") for m in ("pair", "visual") for c in ("baseline", m + "_shuffled")]
    comparisons = []
    table += ["", "| Treatment − control | ΔR@50 | ΔmR@50 | ΔmR@50 seed 42 / 123 / 3407 |", "|---|---:|---:|---|"]
    for a, b in pairs:
        deltas = {k: [100 * (groups[a][s][k] - groups[b][s][k]) for s in (42, 123, 3407)] for k in METRICS}
        means = {k: statistics.mean(v) for k, v in deltas.items()}
        comparisons.append({"treatment": a, "control": b, "paired_delta_pp": deltas, "mean_delta_pp": means})
        table.append(
            f"| {a} − {b} | {means['R@50']:+.3f} | {means['mR@50']:+.3f} | "
            + " / ".join(f"{v:+.3f}" for v in deltas["mR@50"])
            + " |"
        )
    result = {
        "status": "PASS",
        "reports_checked": 27,
        "images_per_report": 26446,
        "plan_sha256": sha256(frozen),
        "summary_sha256": sha256(OUTPUT / "summary.json"),
        "checkpoint_hashes_checked": 27,
        "summary_recomputed": True,
        "gpu_inference_performed": False,
        "comparisons": comparisons,
    }
    outputs = {"analysis.json": json.dumps(result, indent=2) + "\n", "tables.md": "\n".join(table) + "\n"}
    for name, data in outputs.items():
        path = OUTPUT / name
        if path.exists():
            assert path.read_text() == data, f"Changed artifact: {path}"
        else:
            path.write_text(data)
    print("\n".join(table))


if __name__ == "__main__":
    main()
