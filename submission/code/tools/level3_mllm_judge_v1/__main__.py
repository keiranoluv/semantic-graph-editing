import argparse
import uuid

from tools.level3_qwen_unified.core import lock

from .protocol import CONFIG, RELEASE, RUN, atomic, checked, datasets, identity, project_vn100, read, snapshot, versions
from .runner import check_release, execute, protocol_header, release_record, smoke_dir, verified


def main():
    parser = argparse.ArgumentParser(description="Đối chứng Qwen3-VL trên VN100")
    parser.add_argument("command", choices=("prepare", "check", "smoke", "freeze", "score", "report"))
    args = parser.parse_args()
    cfg = read(CONFIG)
    versions()
    if args.command == "prepare":
        print(snapshot(cfg, download=True))
        return
    cases, smoke = datasets(cfg)
    if args.command == "check":
        from transformers import AutoProcessor

        from .backend import encode

        if cases != project_vn100(cfg):
            raise ValueError("Frozen inference manifest differs from VN100 source run")
        folder = snapshot(cfg, weights=False)
        processor = AutoProcessor.from_pretrained(folder, use_fast=True, local_files_only=True)
        for case in smoke:
            encode(processor, checked(cfg["prompt"]).read_text(), cfg, case)
        print("CPU OK: 200 IDs / 100 nguồn, 5 cặp MagicBrush ngoài VN100 qua processor.")
        return
    header = protocol_header(cfg, identity(cfg), cases, smoke)
    if args.command == "smoke":
        folder = smoke_dir(header)
        with lock(folder.parent / ".smoke.lock"):
            if (folder / "oom.json").is_file():
                folder.rename(folder.with_name(folder.name + "-oom-" + uuid.uuid4().hex[:12]))
            rows = execute(folder, smoke, header, cfg, main=False)
        for row in rows:
            print(row["task_id"], row["raw"])
        print(folder, flush=True)
        stats = release_record(header, smoke)["smoke_summary"]
        for kind in ("load", "score"):
            peak = stats[f"max_{kind}_peak_mib"]
            print(f"Peak VRAM {kind} (MiB): allocated={peak['allocated_mib']:.1f}, reserved={peak['reserved_mib']:.1f}")
        print("5/5 JSON hợp lệ. Xem evidence từng cặp trước khi freeze.")
    elif args.command == "freeze":
        record = release_record(header, smoke)
        with lock(RUN.parent / ".freeze.lock"):
            if RELEASE.exists() and read(RELEASE) != record:
                raise ValueError("An existing release is immutable")
            atomic(RELEASE, record)
        print(RELEASE)
    else:
        check_release(header, smoke)
        if args.command == "score":
            rows = execute(RUN, cases, header, cfg, main=True)
            print(f"{sum(r['scores']['status'] == 'ok' for r in rows)}/{len(rows)} JSON hợp lệ. {RUN}")
        else:
            from .report import build, references

            with lock(RUN / ".lock"):
                rows = verified(RUN, cases, header)
                print(build(RUN, rows, references(cfg, cases), cfg, header))


if __name__ == "__main__":
    main()
