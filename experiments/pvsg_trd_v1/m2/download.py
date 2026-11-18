"""Download pinned public PVSG archives, with resume and SHA256 verification."""

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item, directory, *, attempts=4):
    destination = Path(directory) / item["filename"]
    if destination.name != item["filename"]:
        raise ValueError("Archive filename must be a basename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != item["bytes"] or sha256(destination) != item["sha256"]:
            raise ValueError(f"Existing archive failed verification: {destination}")
        return {
            "filename": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": item["sha256"],
            "cached": True,
        }
    partial = destination.with_suffix(destination.suffix + ".partial")
    last_error = None
    for attempt in range(attempts):
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            if offset < item["bytes"]:
                request = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Unexpected resume Content-Range")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with partial.open(mode) as output:
                        while chunk := response.read(8 * 1024 * 1024):
                            output.write(chunk)
            if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
                raise ValueError(f"Downloaded archive size/SHA256 mismatch: {destination.name}")
            partial.replace(destination)
            print(f"Verified {destination.name}: {item['bytes']} bytes", flush=True)
            return {"filename": destination.name, "bytes": item["bytes"], "sha256": item["sha256"], "cached": False}
        except (OSError, ValueError) as error:
            last_error = error
            if isinstance(error, ValueError):
                raise
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Download failed: {destination.name}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifests/media.lock.json")
    parser.add_argument("--directory", type=Path, default=Path("data/external/pvsg_media_v1/archives"))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: download_one(item, args.directory), manifest["archives"]))
    (args.directory / "download_verification.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
