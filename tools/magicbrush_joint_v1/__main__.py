"""Local joint-VLM MagicBrush pipeline. Every stage is explicitly selected."""

import argparse
import json
import os
import shlex
from pathlib import Path

from src.knowledge.transe.core import sha256
from tools.magicbrush_joint_v1 import review
from tools.magicbrush_joint_v1.compatibility import verify_model
from tools.magicbrush_joint_v1.prepare import DEFAULT_CONFIG, config, plan, prepare
from tools.magicbrush_joint_v1.runner import compile_run, infer
from tools.magicbrush_joint_v1.server import runtime, serve, server_args
from tools.magicbrush_joint_v1.storage import exclusive
from tools.prepare_magicbrush import ROOT, encode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "plan",
            "verify-model",
            "prepare",
            "runtime",
            "server-plan",
            "serve-model",
            "infer",
            "compile",
            "review-build",
            "review-serve",
            "review-audit",
            "expand",
        ),
    )
    parser.add_argument("--scope", choices=("pilot", "expanded"), default="pilot")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compilation", type=Path)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args(argv)
    cfg = config(args.config)
    if args.stage in ("review-audit", "expand") and not args.review_csv:
        parser.error("This stage requires a real --review-csv from the viewer")
    result = None
    if args.stage == "plan":
        result = plan(cfg)
    elif args.stage == "verify-model":
        result = verify_model(cfg)
    elif args.stage == "prepare":
        with exclusive(ROOT / cfg["output"]):
            result = prepare(cfg)
    elif args.stage == "runtime":
        result = runtime(cfg)
    elif args.stage == "server-plan":
        result = {"command": shlex.join(server_args(cfg)), "GPU_started": False, "automatic_queue_continuation": False}
    elif args.stage == "serve-model":
        code = serve(cfg)  # ONLY the user explicitly invokes this GPU stage.
        raise SystemExit(code)
    elif args.stage == "infer":
        result = infer(cfg, scope=args.scope, limit=args.limit)
    elif args.stage == "compile":
        path, summary = compile_run(cfg, scope=args.scope)
        output = ROOT / cfg["output"]
        # Mutable convenience pointer only. All evidence/compilations remain immutable.
        pointer = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        with exclusive(output):
            temporary = output / f".pointer-{os.getpid()}.tmp"
            temporary.write_bytes(encode(pointer))
            os.replace(temporary, output / "current-compilation.json")
        result = {"compilation": str(path), **summary}
    elif args.stage == "review-build":
        result = review.build(cfg, args.compilation)
    elif args.stage == "review-serve":
        review.serve(cfg, args.compilation, args.port)
    elif args.stage == "review-audit":
        result = review.audit(cfg, args.review_csv, args.compilation)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ready"] else 2)
    elif args.stage == "expand":
        result = review.expand(cfg, args.review_csv, args.compilation)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
