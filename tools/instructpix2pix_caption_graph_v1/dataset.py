"""Pinned HF Parquet metadata projection with bounded, durable HTTP range caching.

Image bytes are never projected here. The optional image stage uses the viewer
API, verifies captions against the pinned row and records failures explicitly.
"""

import hashlib
import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests

from tools.magicbrush_joint_v1.storage import read, write_once
from tools.prepare_magicbrush import ROOT, digest, fixed, image_record

REPO = "timbrooks/instructpix2pix-clip-filtered"
REVISION = "aa665b890915f7a42f8615bee868a9f3447e178f"
CAPTIONS = ("original_prompt", "edited_prompt", "edit_prompt")
EXPECTED = {
    "original_prompt": "string",
    "original_image": "image",
    "edit_prompt": "string",
    "edited_prompt": "string",
    "edited_image": "image",
}
SERVER = "https://datasets-server.huggingface.co"


def get_json(url, **kwargs):
    response = requests.get(url, timeout=(15, 60), **kwargs)
    response.raise_for_status()
    return response.json()


class RangeFile(io.RawIOBase):
    """Seekable read-only file. A server ignoring Range cannot trigger a full download."""

    def __init__(self, url, size, cache, session=None):
        super().__init__()
        self.url, self.size, self.cache = url, size, Path(cache)
        self.pos = 0
        self.session = session or requests.Session()
        self.network_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if position < 0 or whence not in (0, 1, 2):
            raise ValueError("invalid_range_seek")
        self.pos = position
        return self.pos

    def read(self, length=-1):
        length = self.size - self.pos if length < 0 else min(length, self.size - self.pos)
        if length <= 0:
            return b""
        if length > 8 * 1024 * 1024:
            raise ValueError("range_exceeds_8MiB_metadata_budget; image columns are forbidden")
        start, end = self.pos, self.pos + length - 1
        key = digest([self.url, self.size, start, end])
        target = self.cache / f"{key}.bin"
        receipt = self.cache / f"{key}.json"
        if target.exists() and receipt.exists():
            blob = target.read_bytes()
            if read(receipt) != {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": length} or len(blob) != length:
                raise ValueError("range_cache_corrupt")
        else:
            # Some intermediate caches ignore the Range header in their cache key.
            with self.session.get(
                self.url,
                params={"caption_range": f"{start}-{end}"},
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(15, 60),
            ) as response:
                response.raise_for_status()
                if (
                    response.status_code != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                ):
                    raise ValueError("server_did_not_honor_exact_range")
                blob = response.raw.read(length + 1, decode_content=True)
                if len(blob) != length:
                    raise ValueError("range_length_mismatch")
            self.network_bytes += len(blob)
            fixed(target, blob)
            write_once(receipt, {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": length})
        self.pos += length
        return blob


def parquet(source, shard, cache):
    import pyarrow.parquet as pq

    url = f"https://huggingface.co/datasets/{REPO}/resolve/{source['revision']}/{shard['path']}"
    file = RangeFile(url, shard["size"], Path(cache) / "ranges")
    return pq.ParquetFile(file), file


def inspect(output):
    output = Path(output)
    path = output / "source.json"
    if path.exists():
        source = read(path)
        if source["repo"] != REPO or source["revision"] != REVISION:
            raise ValueError("source_identity_changed")
        return source
    info = get_json(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}")
    fields = info["cardData"]["dataset_info"]["features"]
    if {f["name"]: f["dtype"] for f in fields} != EXPECTED or info["sha"] != REVISION:
        raise ValueError("unexpected_hf_features_or_revision")
    splits = info["cardData"]["dataset_info"]["splits"]
    if len(splits) != 1 or splits[0]["name"] != "train":
        raise ValueError("unexpected_official_splits")
    tree = get_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/data", params={"limit": 1000})
    shards = sorted((x for x in tree if x["path"].endswith(".parquet")), key=lambda x: x["path"])
    if not shards or len(shards) != len([s for s in info["siblings"] if s["rfilename"].endswith(".parquet")]):
        raise ValueError("incomplete_shard_listing")
    card = requests.get(f"https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md", timeout=60)
    card.raise_for_status()
    source = {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "features": fields,
        "splits": splits,
        "shards": shards,
        "dataset_card_sha256": hashlib.sha256(card.content).hexdigest(),
        "license": {
            "card_declaration": "custom license; refer to licensing file",
            "license_tag": info["cardData"].get("license"),
            "license_files": [
                s["rfilename"]
                for s in info["siblings"]
                if any(k in s["rfilename"].lower() for k in ("license", "copying"))
            ],
            "terms_verified": False,
        },
        "full_image_dataset_downloaded": False,
    }
    p, _ = parquet(source, shards[0], output)
    source["physical_schema"] = str(p.schema_arrow)
    if set(p.schema_arrow.names) != set(EXPECTED):
        raise ValueError("unexpected_physical_schema")
    fixed(output / "README.upstream.md", card.content)
    write_once(path, source)
    return source


def plan(output, *, count=100, seed=42, workers=8):
    """Uniform SRS without replacement over *all* pinned rows, before parsing."""
    output = Path(output)
    source = inspect(output)
    index_path = output / "shard-index.json"
    if index_path.exists():
        index = read(index_path)
        if index["source_sha256"] != digest(source):
            raise ValueError("shard_index_source_changed")
    else:

        def describe(shard):
            p, _ = parquet(source, shard, output)
            return {
                "path": shard["path"],
                "size": shard["size"],
                "num_rows": p.metadata.num_rows,
                "schema": str(p.schema_arrow),
            }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            shards = list(pool.map(describe, source["shards"]))
        if any(s["schema"] != source["physical_schema"] for s in shards):
            raise ValueError("heterogeneous_shard_schema")
        index = {"source_sha256": digest(source), "shards": shards, "num_rows": sum(s["num_rows"] for s in shards)}
        if index["num_rows"] != source["splits"][0]["num_examples"]:
            raise ValueError("row_count_card_footer_mismatch")
        write_once(index_path, index)
    if not 1 <= count <= index["num_rows"]:
        raise ValueError("invalid_cohort_size")
    # Shuffle once and take a prefix so expansion retains every diagnostic ID.
    ordering = list(range(index["num_rows"]))
    random.Random(seed).shuffle(ordering)
    wanted = set(ordering[:count])
    locations, offset = {}, 0
    for shard in index["shards"]:
        for row in sorted(i for i in wanted if offset <= i < offset + shard["num_rows"]):
            locations[row] = {
                "global_row": row,
                "shard": shard["path"],
                "row_index": row - offset,
                "shard_size": shard["size"],
            }
        offset += shard["num_rows"]
    result = {
        "schema_version": 1,
        "source_sha256": digest(source),
        "seed": seed,
        "count": count,
        "sampling": "uniform_random_permutation_all_pinned_train_rows_prefix_before_filters",
        "records": [locations[i] for i in ordering[:count]],
        "diagnostic_ids": [identifier(source, locations[i]) for i in ordering[: min(100, count)]],
        "test_policy": "source-family-hash holdout; test captions not parsed or inspected; diagnostic train/val only reported separately",
    }
    write_once(output / "plan.json", result)
    return result


def identifier(source, location):
    return f"ip2p:{source['revision']}:{Path(location['shard']).name}:{location['row_index']}"


def metadata(output, workers=8):
    output = Path(output)
    source, selection = read(output / "source.json"), read(output / "plan.json")
    grouped = {}
    for row in selection["records"]:
        grouped.setdefault(row["shard"], []).append(row)

    def project(items):
        shard, wanted = items
        p, _ = parquet(source, {"path": shard, "size": wanted[0]["shard_size"]}, output)
        # Include image path leaves (tiny metadata), never image bytes. Their
        # stems preserve the upstream sample family when available.
        table = p.read(columns=[*CAPTIONS, "original_image.path", "edited_image.path"], use_threads=False).to_pylist()
        records = []
        for location in wanted:
            row = table[location["row_index"]]
            paths = {
                r: row.get(r, {}).get("path") if isinstance(row.get(r), dict) else None
                for r in ("original_image", "edited_image")
            }
            records.append(
                {
                    "id": identifier(source, location),
                    **location,
                    **{k: row.get(k) for k in CAPTIONS},
                    "upstream_image_paths": paths,
                    "provenance": {
                        "repo": REPO,
                        "revision": source["revision"],
                        "official_split": "train",
                        "source_sha256": digest(source),
                    },
                    "image_availability": "not_requested",
                }
            )
        return records

    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(project, grouped.items()))
    records = {r["id"]: r for b in batches for r in b}
    result = [records[identifier(source, r)] for r in selection["records"]]
    write_once(output / "metadata.json", {"plan_sha256": digest(selection), "records": result})
    return result


def prepare_images(output, records, *, retry=False):
    """Optional per-row image fetch. No full-shard fallback on API scan errors."""
    output = Path(output)
    receipts = output / "image-receipts"
    result = {}
    rate_limited = False
    for row in records:
        receipt = receipts / f"{digest(row['id'])}.json"
        resolved = receipts / f"{digest(row['id'])}-resolved.json"
        selected = resolved if resolved.exists() else receipt
        if selected.exists() and (not retry or read(selected)["status"] == "available"):
            value = read(selected)
            for role in ("source_image", "target_image"):
                if role in value:
                    blob = (ROOT / value[role]["path"]).read_bytes()
                    if hashlib.sha256(blob).hexdigest() != value[role]["sha256"]:
                        raise ValueError("cached_image_changed")
            result[row["id"]] = value
            continue
        value = {
            "id": row["id"],
            "status": "unavailable",
            "source_identity": row["provenance"],
            "method": "viewer_row_caption_verified_against_pinned_parquet",
        }
        try:
            if rate_limited:
                raise ValueError("image_download_deferred_after_rate_limit; retry explicitly later")
            response = requests.get(
                f"{SERVER}/rows",
                params={
                    "dataset": REPO,
                    "config": "default",
                    "split": "train",
                    "offset": row["global_row"],
                    "length": 1,
                },
                timeout=(15, 60),
            )
            if not response.ok:
                rate_limited = response.status_code == 429
                raise ValueError(f"viewer_http_{response.status_code}:{response.text[:700]}")
            payload = response.json()["rows"]
            if len(payload) != 1 or payload[0]["row_idx"] != row["global_row"]:
                raise ValueError("viewer_row_identity_mismatch")
            remote = payload[0]["row"]
            if any(remote[k] != row[k] for k in CAPTIONS):
                raise ValueError("viewer_revision_or_caption_mismatch")
            for remote_role, role in (("original_image", "source_image"), ("edited_image", "target_image")):
                url = remote[remote_role]["src"]
                parsed = urlparse(url)
                if parsed.scheme != "https" or parsed.hostname != "datasets-server.huggingface.co":
                    raise ValueError("unexpected_image_host")
                with requests.get(url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    blob = response.raw.read(16 * 1024 * 1024 + 1, decode_content=True)
                if len(blob) > 16 * 1024 * 1024:
                    raise ValueError("individual_image_exceeds_16MiB")
                value[role] = image_record(blob, output)
            value["status"] = "available"
        except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as error:
            value["error"] = str(error)
        # Retry writes a new content-addressed attempt, never overwrites evidence.
        if receipt.exists():
            receipt = (
                resolved if value["status"] == "available" else receipts / f"{digest(row['id'])}-{digest(value)}.json"
            )
        write_once(receipt, value)
        result[row["id"]] = value
    return result
