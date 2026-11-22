import argparse
import json
import time
from pathlib import Path

from task3_vn100.evaluation.core import (
    DEFAULT_OUTPUT,
    case_header,
    identity,
    inputs,
    read_stage,
    save_stage,
    stage_binding,
    workspace,
)
from task3_vn100.qwen_unified.core import GPU_LOCK, atomic, lock


def main():
    parser = argparse.ArgumentParser(
        description="Independent evaluation of the frozen 200 Qwen pairs; never generates images or SGs."
    )
    parser.add_argument("command", choices=["check", "graph", "metrics"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true", help="One real pair; separate non-evaluation output.")
    args = parser.parse_args()
    cfg, records = inputs()
    if args.command == "check":
        from task3_vn100.evaluation.metrics import check_assets, check_captions

        print(json.dumps({"pairs": len(records), "metric_assets": len(check_assets()), **check_captions(records)}))
        return
    code = identity()
    out = workspace(args.output, cfg, code)
    with lock(out / ".evaluation.lock"):
        if args.command == "graph":
            from task3_vn100.evaluation.scoring import score, summarize

            path = out / "graph/results.json"
            binding = stage_binding(cfg, records)
            if path.exists():
                from task3_vn100.evaluation.validation import validate_stage

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
            from task3_vn100.evaluation.metrics import run

            if args.device == "cuda":
                with lock(GPU_LOCK):
                    result = run(out, cfg, records, args.device, args.smoke)
            else:
                result = run(out, cfg, records, args.device, args.smoke)
            print(json.dumps(result))


if __name__ == "__main__":
    main()
