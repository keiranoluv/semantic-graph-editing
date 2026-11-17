"""Independent CPU audit, paired deltas and interpretation of completed test analysis."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_level1 import load_model
from tools.analyze_level1_long_tail import encode, stats, write_fixed
from tools.level1_test_tail_zero_v1 import OUTPUT, VOCAB, infer, load_inputs, sha256, verify_report


def main():
    torch.set_num_threads(1)
    freeze = json.loads((OUTPUT / "freeze.json").read_text())
    cohort = json.loads((OUTPUT / "test-cohort.json").read_text())
    summary = json.loads((OUTPUT / "zero-shot-summary.json").read_text())
    seen = {tuple(t) for t in json.loads(VOCAB.read_text())["triples"]}
    lookup, probes = {}, []
    for entry in freeze["entries"]:
        path = OUTPUT / f"{entry['condition']}-seed{entry['seed']}.json"
        row = json.loads(path.read_text())
        verify_report(row, entry, freeze, cohort)
        if sha256(path) != summary["reports"][path.name]:
            raise ValueError("Summary source hash differs")
        values = np.asarray(row["images"], dtype=np.int64)
        independent = np.mean(values[:, 2:] / values[:, 1:2], axis=0)
        if not np.allclose(independent, list(row["metrics"].values()), rtol=0, atol=1e-12):
            raise ValueError("NumPy audit disagrees with image-macro")
        checkpoint, dataset = load_inputs(entry)
        model = load_model(checkpoint).eval()
        cpu_hits = infer(model, dataset, cohort["images"][:2], seen, "cpu", 1)
        if cpu_hits != row["images"][:2]:
            raise ValueError("CPU/GPU prediction hits differ on probe images")
        probes.append({"condition": entry["condition"], "seed": entry["seed"], "cpu_gpu_hits_equal": True})
        del checkpoint, dataset, model
        lookup[entry["condition"], entry["seed"]] = row["metrics"]
    for condition, metrics in summary["groups"].items():
        for key, saved in metrics.items():
            values = [lookup[condition, seed][key] for seed in (42, 123, 3407)]
            if not np.isclose(np.mean(values), saved["mean"], rtol=0, atol=1e-12):
                raise ValueError("Summary mean differs")
            if not np.isclose(np.std(values, ddof=1), saved["sample_std"], rtol=0, atol=1e-12):
                raise ValueError("Summary sample std differs")
    deltas = {}
    for control in ("fusion_baseline", "fusion_pair_shuffled"):
        deltas[control] = {
            key: stats([100 * (lookup["fusion_pair_real", s][key] - lookup[control, s][key]) for s in (42, 123, 3407)])
            for key in summary["groups"]["fusion_pair_real"]
        }
    result = {
        "status": "PASS",
        "reports": 9,
        "images_per_run": len(cohort["images"]),
        "split": "test",
        "numpy_recomputed": True,
        "cpu_gpu_probes": probes,
        "paired_deltas_pp": deltas,
        "analysis_sha256": sha256(Path(__file__)),
        "summary_sha256": sha256(OUTPUT / "zero-shot-summary.json"),
    }
    write_fixed(OUTPUT / "analysis.json", encode(result))
    lines = [
        "# Phân tích bổ sung VG150 test: long-tail và zero-shot",
        "",
        "Long-tail được tính từ 54 report test có sẵn, không inference lại.",
        "Zero-shot chỉ chạy thêm 9 checkpoint fusion đã khóa; mỗi lượt inference 4.660 ảnh có GT zero-shot.",
        "Cohort thuộc toàn bộ test 26.446 ảnh; 7.848 annotation, không lọc prediction trước top-K.",
        "",
        "[Bảng long-tail](long-tail.md) · [Bảng zero-shot](zero-shot.md) · [Audit](analysis.json)",
        "",
        "## Zero-shot: pair KG thật trừ đối chứng",
        "",
        "Delta theo seed 42/123/3407, đơn vị điểm phần trăm. Không phải kiểm định ý nghĩa thống kê.",
        "",
        "| Đối chứng | Metric | Mean delta | Seed 42 / 123 / 3407 |",
        "|---|---|---:|---|",
    ]
    for control, metrics in deltas.items():
        for key, v in metrics.items():
            lines.append(
                f"| {control} | {key} | {v['mean']:+.3f} | " + " / ".join(f"{x:+.3f}" for x in v["per_seed"]) + " |"
            )
    tail = json.loads((OUTPUT / "long-tail.json").read_text())
    lines += [
        "",
        "## Long-tail: adjusted-off trừ CE-off",
        "",
        "| Nhóm | Delta mR@50 (pp) | Seed 42 / 123 / 3407 |",
        "|---|---:|---|",
    ]
    for group in ("head", "body", "tail"):
        a = tail["conditions"]["loss_adjusted_off"][group]["mR@50"]["per_seed"]
        b = tail["conditions"]["loss_ce_off"][group]["mR@50"]["per_seed"]
        diff = 100 * (np.asarray(a) - np.asarray(b))
        lines.append(f"| {group} | {diff.mean():+.3f} | " + " / ".join(f"{x:+.3f}" for x in diff) + " |")
    lines += [
        "",
        "## Kiểm chứng và phạm vi",
        "",
        "- 29 CPU tests pass trước commit/GPU; smoke checkpoint thật so batch 1 và 2.",
        "- Audit độc lập NumPy từ hits/support của cả 9 lượt, khớp 36 metric và mean/sample std.",
        "- Hai ảnh đầu của từng checkpoint được đối chiếu CPU với kết quả GPU đã lưu; cả 9 khớp hits.",
        "- Train vocabulary và head/body/tail giữ nguyên; checkpoint chọn bằng validation legacy, không chọn lại.",
        "- Zero-shot chỉ khảo sát ba cấu hình fusion, không khẳng định đã chạy 54 checkpoint.",
        "- Ảnh không có GT zero-shot không vào mẫu số zR; mỗi ảnh có GT được trọng số bằng nhau.",
        "- Shuffled dùng một permutation; ba seed cùng dấu chưa chứng minh ý nghĩa thống kê.",
        "- Test đã được xem trước đó; không tuyên bố holdout mới. Artifact validation cũ giữ nguyên.",
        "- Không có optimizer, training hoặc W&B trong lượt inference bổ sung này.",
        "",
    ]
    write_fixed(OUTPUT / "README.md", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
