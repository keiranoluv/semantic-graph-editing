"""Paired-image annotation API and offline ChatGPT JSON exchange."""

import argparse
import io
import json
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from tools.emnlp2020_graph_modification_v1.dataset import encode
from tools.magicbrush_manual_v1.store import MANIFEST, Conflict, validate_pair
from tools.magicbrush_manual_v2.proposal import parse, prompt
from tools.magicbrush_manual_v2.store import LEGACY_OUTPUT, OUTPUT, Store

ASSETS = Path(__file__).with_name("assets")


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, kind="application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            request = urlsplit(self.path)
            query = parse_qs(request.query)
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
            }
            try:
                if request.path in assets:
                    name, kind = assets[request.path]
                    return self.send(200, (ASSETS / name).read_bytes(), kind)
                if request.path == "/api/bootstrap":
                    return self.send(200, encode(store.bootstrap()))
                if request.path == "/api/case":
                    return self.send(200, encode(store.case(query["id"][0])))
                if request.path == "/api/prompt":
                    identifier = query["id"][0]
                    store.directory(identifier)
                    return self.send(200, encode({"prompt": prompt(store.records[identifier])}))
                if request.path == "/api/package":
                    identifier = query["id"][0]
                    store.directory(identifier)
                    content = io.BytesIO()
                    with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for i, role in enumerate(("source", "target"), 1):
                            data, suffix = store.image(identifier, role)
                            archive.writestr(f"{i}-{role}{suffix}", data)
                        archive.writestr("prompt.txt", prompt(store.records[identifier]))
                        archive.writestr(
                            "provenance.json",
                            encode(
                                {
                                    "sample_id": identifier,
                                    "source_manifest_sha256": store.identity["source_manifest_sha256"],
                                    "images": {
                                        r: store.records[identifier][r + "_image"] for r in ("source", "target")
                                    },
                                    "source": "osunlp/MagicBrush",
                                }
                            ),
                        )
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", 'attachment; filename="magicbrush-pair.zip"')
                    self.send_header("Content-Length", str(len(content.getvalue())))
                    self.end_headers()
                    self.wfile.write(content.getvalue())
                    return
                if request.path == "/api/image":
                    content, suffix = store.image(query["id"][0], query["role"][0])
                    kind = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".webp": "image/webp",
                    }.get(suffix)
                    if kind is None:
                        raise ValueError("Định dạng ảnh không hỗ trợ.")
                    return self.send(200, content, kind)
                return self.send(404, encode({"error": "Không có trang này."}))
            except (ValueError, KeyError, OSError) as error:
                return self.send(400, encode({"error": str(error)}))

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin != f"http://{self.headers.get('Host')}" or self.headers.get("Content-Type") != "application/json":
                return self.send(403, encode({"error": "Yêu cầu phải đến từ giao diện đang mở."}))
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 300_000:
                    raise ValueError("Dữ liệu gửi quá lớn hoặc rỗng.")
                body = json.loads(self.rfile.read(size))
                if self.path == "/api/save":
                    return self.send(200, encode(store.save(body["id"], body["action"], body["payload"])))
                if self.path == "/api/validate":
                    current = store.case(body["id"])["state"]
                    return self.send(
                        200,
                        encode(validate_pair(body.get("source_graph", current["source_graph"]), body["target_graph"])),
                    )
                if self.path == "/api/import-preview":
                    store.directory(body["id"])
                    return self.send(200, encode(parse(body["text"], body["id"])))
                if self.path == "/api/import-json":
                    return self.send(
                        200,
                        encode(
                            store.import_json(
                                body["id"], body["text"], body["base_revision"], body.get("annotator", "")
                            )
                        ),
                    )
                if self.path == "/api/export":
                    return self.send(200, encode(store.compile()))
                return self.send(404, encode({"error": "Không có thao tác này."}))
            except Conflict as error:
                return self.send(409, encode({"error": str(error)}))
            except (ValueError, KeyError, TypeError, OSError) as error:
                return self.send(400, encode({"error": str(error)}))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "serve", "compile"), default="serve")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--legacy-output", type=Path, default=LEGACY_OUTPUT)
    parser.add_argument("--port", type=int, default=8776)
    args = parser.parse_args(argv)
    store = Store(args.manifest, args.output, legacy_output=args.legacy_output)
    if args.stage == "plan":
        result = store.bootstrap()
        print(encode({k: v for k, v in result.items() if k != "cases"}).decode(), end="")
        print(
            f"{len(result['cases'])} cặp train/val; hiện hai ảnh ngay từ đầu; {result['migrated_cases']} mẫu được giữ từ v1."
        )
    elif args.stage == "compile":
        print(encode(store.compile()["report"]).decode(), end="")
    else:
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler(store)) as server:
            print(f"MagicBrush hai ảnh + JSON v2: http://127.0.0.1:{server.server_port}/", flush=True)
            print(f"Lưu tại {store.output}; chỉ CPU, không gọi model; Ctrl+C để dừng.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0
