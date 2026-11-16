import argparse
import functools
import http.server
import json
from pathlib import Path

from tools.level3_image_graph_v1.core import read, sha256

from .core import DEFAULT_OUTPUT, prepare


def main():
    p = argparse.ArgumentParser(description="Recheck user-labeled failures without revealing model scores.")
    p.add_argument("command", choices=["prepare", "serve"])
    p.add_argument("--review", default="qwen200-reviewed.json")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--blank-is-pass", action="store_true", help="Only use with explicit user authorization.")
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.review, args.output, args.blank_is_pass), ensure_ascii=False))
        return
    receipt = read(args.output / "receipt.json")
    for name, expected in receipt["files_sha256"].items():
        if sha256(args.output / name) != expected:
            raise ValueError("recheck_artifact_changed:" + name)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.output))
    )
    print(f"http://localhost:{server.server_port}/comparison.html", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
