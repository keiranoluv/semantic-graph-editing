"""Download only the locked official BF16 checkpoint; never invoke inference."""

import requests

from tools.level3_image_graph_v1.core import read

from .core import CACHE, LOCK, assets


def prepare():
    for f in read(LOCK)["files"]:
        p = CACHE / "model" / f["path"]
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(f["url"], stream=True, timeout=180) as response:
            response.raise_for_status()
            with p.open("xb") as out:
                for block in response.iter_content(8 * 1024 * 1024):
                    out.write(block)
    return assets()
