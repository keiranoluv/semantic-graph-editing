"""Post-hoc validation experiment for seven runs stopped at the first patience boundary."""

import argparse
import fcntl
import os
from contextlib import ExitStack
from pathlib import Path

import torch

from extension_pvsg.contracts import read_json
from extension_pvsg.m2.download import sha256
from extension_pvsg.m3.common import PROJECT, write_json
from extension_pvsg.m3.engine import should_stop
from extension_pvsg.m3.runner import train as train_m3
from extension_pvsg.m3_150.queue import setup
from extension_pvsg.m4.common import require_committed, verify_parent, verify_self
from extension_pvsg.m4.preflight import check_profile
from extension_pvsg.m4.runner import train as train_m4
from extension_pvsg.m4_150_bs32.run import settings
from extension_pvsg.validate_m0 import canonical_sha256

ROOT = Path("reports/pvsg-trd-min50-followup-v1")
CASES = {"st_gat": (17, 2026), "st_no_temporal": (17, 2026), "st_no_edges": (17, 42, 2026)}
NOTE = (
    "Kiểm tra bổ sung được chọn sau khi xem validation: chỉ chạy lại các lượt dừng ở epoch 16 "
    "với R/mR@50 bằng 0. Huấn luyện lại từ đầu với cùng seed để kiểm tra hội tụ muộn. "
    "Early stopping có hiệu lực từ cuối epoch 50, patience 15 tính liên tục, tối đa 150 epoch. "
    "Bảng chính dùng toàn bộ kết quả của protocol ban đầu."
)


def prepare():
    base, digest, m3_profile, _ = setup()
    require_committed()
    verify_parent()
    verify_self()
    cases = []
    for model, seeds in CASES.items():
        m3 = model == "st_gat"
        cfg = base if m3 else settings()
        original = Path(cfg["output_root"])
        profile = m3_profile if m3 else check_profile(original / "profiles" / f"{model}.json", cfg, digest, model)
        for seed in seeds:
            path = original / model / f"seed-{seed}"
            done = read_json(path / "complete.json")
            ident = read_json(path / "run.json")["identity"]
            if (
                ident["config"] != cfg
                or ident["profile_sha256"] != canonical_sha256(profile)
                or ident["batch_size"] != 32
                or ident["feature_summary_sha256"] != digest
                or not done["early_stopped"]
                or done["epochs"] != 16
                or done["best_key"] != [0, 0]
                or done["history_sha256"] != sha256(path / "history.json")
            ):
                raise ValueError(f"Unexpected baseline: {path}")
            followup = dict(
                cfg,
                min_epochs=50,
                output_root=str(ROOT),
                experiment_id=ROOT.name,
                baseline_signature=done["signature"],
                diagnostic_note=NOTE,
                followup_launcher_sha256=sha256(__file__),
            )
            assert not should_stop(49, 48, followup) and should_stop(50, 49, followup)
            cases.append((model, seed, followup, profile, digest, path))
    return cases


def report(cases):
    rows = []
    for model, seed, _, _, _, baseline in cases:
        out = ROOT / model / f"seed-{seed}"
        orig = read_json(baseline / "complete.json")
        row = dict(
            model=model,
            seed=seed,
            baseline_path=str(baseline),
            baseline_epochs=orig["epochs"],
            baseline_best_mR50_R50=orig["best_key"],
            status="pending",
        )
        if (out / "history.json").exists():
            h = read_json(out / "history.json")
            best = max(h, key=lambda x: x["selection_key"])
            row.update(
                status="running",
                epochs=h[-1]["epoch"],
                best_epoch=best["epoch"],
                best_mR50_R50=best["selection_key"],
                first_positive_R50_epoch=next((x["epoch"] for x in h if x["selection_key"][1] > 0), None),
                best_mR50_R50_through_epoch16=max(x["selection_key"] for x in h if x["epoch"] <= 16),
            )
            if (out / "complete.json").exists():
                done = read_json(out / "complete.json")
                if done["epochs"] < 50 or done["best_key"] != best["selection_key"]:
                    raise ValueError(f"Inconsistent follow-up: {out}")
                row["status"] = "complete"
        rows.append(row)
    write_json(ROOT / "validation_followup.json", dict(note=NOTE, metric="internal validation %, tIoU=0.5", cases=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "train", "report"))
    args = parser.parse_args()
    os.chdir(PROJECT)
    cases = prepare()
    if args.action == "check":
        print("Verified original runs:", [(m, s) for m, s, *_ in cases])
        return
    if args.action == "report":
        report(cases)
        return
    with ExitStack() as stack:
        for path in ("reports/pvsg-trd-m3-v1/.gpu.lock", "reports/pvsg-trd-m4-150-v1/.gpu.lock"):
            lock = stack.enter_context(open(path, "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device("cuda:0")
        report(cases)
        for model, seed, cfg, profile, digest, _ in cases:
            if torch.cuda.get_device_name(device) != profile["gpu"]["name"] or not torch.cuda.is_bf16_supported():
                raise ValueError("Profile GPU or BF16 mismatch")
            torch.cuda.set_per_process_memory_fraction(cfg.get("allocator_fraction", 1.0), device)
            row = next(x for x in profile["models"][model]["candidates"] if x["batch_size"] == 32)
            if torch.cuda.mem_get_info(device)[0] < 1.1 * max(
                row[k]["peak_reserved_bytes"] for k in ("heavy", "mixed")
            ):
                raise ValueError("Insufficient free VRAM for batch 32")
            trainer = train_m3 if model == "st_gat" else train_m4
            trainer(cfg, model, seed, profile, digest, device)
            report(cases)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
