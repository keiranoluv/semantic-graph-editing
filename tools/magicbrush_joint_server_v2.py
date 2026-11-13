"""Server-only entrypoint for the frozen joint-image v1 annotation pipeline.

Run runtime/server-plan with CPU only. serve-model is explicitly user-invoked.
The v1 all-stage CLI imports optional human-review/GQA dependencies eagerly;
this separate, versioned entrypoint keeps those imports out of the server env.
"""

import argparse
import json
import shlex
from pathlib import Path


def launcher_provenance(cfg):
    from src.knowledge.transe.core import sha256
    from tools.magicbrush_joint_v1.prepare import check_run, revision
    from tools.prepare_magicbrush import ROOT, digest

    run, _ = check_run(cfg)
    source = Path(__file__).resolve()
    return {
        "schema_version": "magicbrush_joint_server_entrypoint_v2",
        "entrypoint": str(source.relative_to(ROOT)),
        "entrypoint_sha256": sha256(source),
        "entrypoint_code_revision": revision(),
        "annotation_run_sha256": digest(run),
        "annotation_code_revision": run["code_revision"],
        "model": cfg["model"],
        "batch_size": cfg["batch_size"],
        "seed": cfg["seed"],
        "automatic_queue_continuation": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", required=True, choices=("runtime", "server-plan", "serve-model"))
    args = parser.parse_args(argv)

    # Parse --help before importing anything beyond the standard library.
    # Deliberately avoid the annotation runner, review module and all-stage CLI.
    from tools.magicbrush_joint_v1.prepare import DEFAULT_CONFIG, config
    from tools.magicbrush_joint_v1.server import runtime, serve, server_args

    cfg = config(args.config or DEFAULT_CONFIG)
    if args.stage == "runtime":
        result = runtime(cfg)
    elif args.stage == "server-plan":
        result = {
            "command": shlex.join(server_args(cfg)),
            "GPU_started": False,
            "automatic_queue_continuation": False,
        }
    else:
        from tools.magicbrush_joint_v1.storage import exclusive, read, write_once
        from tools.prepare_magicbrush import ROOT

        provenance = launcher_provenance(cfg)
        output = ROOT / cfg["output"]
        path = output / "launchers/server-entrypoint-v2.json"
        with exclusive(output):
            if path.exists():
                previous = read(path)
                # Unrelated later commits do not change the frozen entrypoint source.
                provenance["entrypoint_code_revision"] = previous["entrypoint_code_revision"]
            write_once(path, provenance)
        raise SystemExit(serve(cfg))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
