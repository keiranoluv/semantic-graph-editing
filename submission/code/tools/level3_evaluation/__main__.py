import argparse
import json
import time
from pathlib import Path

from tools.level3_qwen_unified.core import GPU_LOCK, atomic, lock

from .core import DEFAULT_OUTPUT, case_header, identity, inputs, read_stage, save_stage, stage_binding, workspace


def main():
    parser = argparse.ArgumentParser(
        description="Independent evaluation of the frozen 200 Qwen pairs; never generates images or SGs."
    )
    parser.add_argument("command", choices=["check", "graph", "metrics", "report"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One real pair; separate non-evaluation output.")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cfg, records = inputs()
    if args.command == "check":
        from .metrics import check_assets, check_captions

        print(json.dumps({"pairs": len(records), "metric_assets": len(check_assets()), **check_captions(records)}))
        return
    code = identity()
    out = workspace(args.output, cfg, code)
    with lock(out / ".evaluation.lock"):
        if args.command == "graph":
            from .scoring import score, summarize

            path = out / "graph/results.json"
            binding = stage_binding(cfg, records)
            if path.exists():
                from .report import validate_stage

                pack = read_stage(path)
                validate_stage(pack, records, binding)
                summary = summarize(pack["rows"])
                summary["resumed_existing"] = True
            else:
                started = time.perf_counter()
                rows = [{**case_header(r), "scores": score(r, cfg["graph_config"])} for r in records]
                save_stage(path, {"stage_binding": binding, "evaluation_eligible": True, "rows": rows})
                summary = summarize(rows)
                summary["seconds"] = time.perf_counter() - started
                atomic(out / "graph/summary.json", summary)
            print(json.dumps({"output": str(path.parent), **summary}))
        elif args.command == "metrics":
            from .metrics import run

            if args.device == "cuda":
                with lock(GPU_LOCK):
                    result = run(out, cfg, records, args.device, args.smoke)
            else:
                result = run(out, cfg, records, args.device, args.smoke)
            print(json.dumps(result))
        else:
            from .report import run

            if args.review is None:
                parser.error("report requires --review /path/to/qwen200-reviewed.json")
            print(json.dumps(run(out, cfg, records, args.review, args.allow_partial)))


if __name__ == "__main__":
    main()
