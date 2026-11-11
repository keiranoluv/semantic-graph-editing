"""Download one Visual Genome JPEG by image ID for the optional level 1 demo."""

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-id", type=int, required=True)
    p.add_argument("--root", type=Path, default=Path("data/raw/vg150"))
    args = p.parse_args(argv)
    if args.image_id <= 0:
        p.error("image-id must be positive")
    records = json.loads((args.root / "image_data.json").read_text(encoding="utf-8"))
    matches = [row for row in records if row.get("image_id") == args.image_id]
    if len(matches) != 1:
        raise ValueError("Image ID missing or duplicated in aligned metadata")
    url = matches[0]["url"]
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "cs.stanford.edu"
        or not parsed.path.startswith("/people/rak248/VG_100K")
    ):
        raise ValueError("Unexpected image source URL; inspect metadata and source manually")
    target = args.root / "images" / f"{args.image_id}.jpg"
    if target.exists():
        print(f"[SKIP] Existing image: {target}")
        return
    try:
        with urlopen(url, timeout=30) as response:
            data = response.read(20_000_001)
    except URLError as exc:
        raise RuntimeError(
            "Visual Genome image host is unavailable from this machine; "
            "see the official Visual Genome image archives or use the notebook without JPEG"
        ) from exc
    if len(data) > 20_000_000 or not data.startswith(b"\xff\xd8"):
        raise ValueError("Image is too large or is not a JPEG")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(data)
    print(f"[OK] {target} ({len(data)} bytes) from {url}")


if __name__ == "__main__":
    main()
