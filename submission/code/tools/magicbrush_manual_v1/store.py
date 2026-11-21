"""Append-only manual annotations with a source-first workflow and CAS writes."""

import copy
import fcntl
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.editing.operations import KINDS, EditGraph, apply_operations, diff_aligned_graphs
from tools.edit_model_v1.codec import canonical_source, decode_actions, encode_actions, normalize_label
from tools.emnlp2020_graph_modification_v1.dataset import digest, read, sha256, write
from tools.magicbrush_manual_v1 import VERSION

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/processed/level2/magicbrush-v1/manifest.json"
OUTPUT = ROOT / "data/processed/level2/magicbrush-manual-v1"
EMPTY = {"nodes": [], "edges": []}


class Conflict(ValueError):
    pass


def code_hashes():
    files = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).with_name("assets").glob("*"))
    files += [ROOT / "src/editing/operations.py", ROOT / "tools/edit_model_v1/codec.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def graph(payload):
    """Drafts retain incomplete text; structural shape and size remain bounded."""
    if not isinstance(payload, dict) or set(payload) != {"nodes", "edges"}:
        raise ValueError("Graph cần danh sách đối tượng và quan hệ.")
    if not isinstance(payload["nodes"], list) or not isinstance(payload["edges"], list):
        raise ValueError("Danh sách graph không hợp lệ.")
    if len(payload["nodes"]) > 100 or len(payload["edges"]) > 200:
        raise ValueError("Bản nháp vượt giới hạn lưu trữ.")
    for kind, fields in (("nodes", {"id", "label", "attributes"}), ("edges", {"id", "subject", "predicate", "object"})):
        ids = []
        for row in payload[kind]:
            if not isinstance(row, dict) or set(row) != fields:
                raise ValueError("Trường đối tượng/quan hệ không hợp lệ.")
            for key, value in row.items():
                if key == "attributes":
                    if (
                        not isinstance(value, list)
                        or len(value) > 30
                        or any(not isinstance(a, str) or len(a) > 120 for a in value)
                    ):
                        raise ValueError("Thuộc tính cần là các cụm từ ngắn.")
                elif not isinstance(value, str) or len(value) > 160:
                    raise ValueError("Nhãn/quan hệ quá dài hoặc không hợp lệ.")
            ids.append(row["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("ID đối tượng/quan hệ bị trùng.")
    return copy.deepcopy(payload)


def points(payload, g):
    if not isinstance(payload, dict) or set(payload) - {n["id"] for n in g["nodes"]}:
        raise ValueError("Điểm đánh dấu không thuộc đối tượng trong graph.")
    for xy in payload.values():
        if not isinstance(xy, dict) or set(xy) != {"x", "y"}:
            raise ValueError("Điểm đánh dấu không hợp lệ.")
        if any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in xy.values()):
            raise ValueError("Điểm đánh dấu nằm ngoài ảnh.")
    return copy.deepcopy(payload)


def parse_graph(payload):
    payload = graph(payload)
    for i, node in enumerate(payload["nodes"], 1):
        if not node["label"].strip():
            raise ValueError(f"Đối tượng {i} chưa có tên.")
        if len(set(node["attributes"])) != len(node["attributes"]):
            raise ValueError(f"Thuộc tính của {node['label']} đang bị lặp.")
    for i, edge in enumerate(payload["edges"], 1):
        if not edge["predicate"].strip():
            raise ValueError(f"Quan hệ {i} chưa được điền.")
        if any(edge[key] not in {n["id"] for n in payload["nodes"]} for key in ("subject", "object")):
            raise ValueError(f"Quan hệ {i} trỏ đến vật thể không còn trong ảnh.")
    return EditGraph.from_dict(payload)


def validate_pair(source, target):
    result = {"valid": False, "problems": [], "operations": None, "operation_counts": None}
    try:
        source, target = parse_graph(source), parse_graph(target)
        if not source.nodes or not target.nodes:
            raise ValueError("Mỗi ảnh cần ít nhất một đối tượng nhìn thấy.")
        if len(source.nodes) > 20 or len(target.nodes) > 20 or len(source.edges) > 60 or len(target.edges) > 60:
            raise ValueError("Vượt budget: tối đa 20 đối tượng và 60 quan hệ mỗi ảnh.")
        old_nodes, old_edges = {n.id for n in source.nodes}, {e.id for e in source.edges}
        if any(n.id not in old_nodes and not re.fullmatch(r"newn[0-9]{3,}", n.id) for n in target.nodes):
            raise ValueError("Đối tượng mới phải được thêm bằng nút Thêm đối tượng.")
        if any(e.id not in old_edges and not re.fullmatch(r"newe[0-9]{3,}", e.id) for e in target.edges):
            raise ValueError("Quan hệ mới phải được thêm bằng nút Thêm quan hệ.")
        ops = diff_aligned_graphs(source, target)
        if apply_operations(source, ops) != target:
            raise ValueError("Thao tác không tái dựng đúng graph đích.")
        source, target, ops = normalize_label(source, target, ops)
        result["operations"] = [o.to_dict() for o in ops]
        result["operation_counts"] = {k: sum(o.kind == k for o in ops) for k in KINDS}
        result["source_graph"], result["target_graph"] = source.to_dict(), target.to_dict()
        if not 1 <= len(ops) <= 12:
            raise ValueError(
                "Hai graph chưa có thay đổi." if not ops else "Vượt budget 12 thao tác; giữ bản nháp và ghi chú."
            )
        tokens = encode_actions(ops)
        if len(tokens) - 1 > 2048:
            raise ValueError("Nhãn thao tác vượt budget 2048 token.")
        if apply_operations(source, decode_actions(tokens[1:], source)) != target:
            raise ValueError("Codec không tái dựng đúng graph đích.")
        result["valid"] = True
    except (ValueError, TypeError, KeyError) as error:
        result["problems"].append(str(error))
    return result


class Store:
    def __init__(self, manifest=MANIFEST, output=OUTPUT, *, root=ROOT):
        self.root, self.manifest_path, self.output = (
            Path(root).resolve(),
            Path(manifest).resolve(),
            Path(output).resolve(),
        )
        manifest = read(self.manifest_path)
        self.original = manifest
        if manifest.get("dataset") != "osunlp/MagicBrush":
            raise ValueError("Cần manifest MagicBrush có provenance ảnh.")
        records = manifest["records"]
        if len({r["id"] for r in records}) != len(records):
            raise ValueError("Manifest trùng ID.")
        seen, groups, pixels = {}, {}, {}
        index = {r["id"]: r for r in records}
        for split in ("train", "val", "test"):
            for identifier in manifest["splits"][split]:
                if identifier in seen or identifier not in index:
                    raise ValueError("Split trùng/không có ID.")
                row = index[identifier]
                if row["split"] != split or row["official_split"] != "train":
                    raise ValueError("Split hoặc provenance không khớp.")
                seen[identifier] = split
                for key in (row["source_group_id"], "session:" + row["img_id"]):
                    if groups.setdefault(key, split) != split:
                        raise ValueError("Rò rỉ source group/session qua split.")
                for role in ("source_image", "target_image"):
                    if pixels.setdefault(row[role]["pixel_sha256"], split) != split:
                        raise ValueError("Rò rỉ ảnh qua split.")
                    path = (self.root / row[role]["path"]).resolve()
                    if not path.is_relative_to(self.root / "data"):
                        raise ValueError("Đường dẫn ảnh ngoài thư mục data.")
        if set(seen) != set(index):
            raise ValueError("Split thiếu record.")
        self.records = {i: r for i, r in index.items() if r["split"] in ("train", "val")}
        self.review_ids = manifest["review_ids"]
        if (
            len(self.review_ids) != 20
            or len(set(self.review_ids)) != 20
            or any(i not in self.records or self.records[i]["split"] != "train" for i in self.review_ids)
        ):
            raise ValueError("Cần giữ nguyên 20 ID review train đã chọn trước.")
        self.identity = {
            "schema": VERSION,
            "source_manifest_sha256": sha256(self.manifest_path),
            "source_dataset_revision": manifest["revision"],
            "splits": ["train", "val"],
            "review_ids": self.review_ids,
            "test_images_exposed": False,
            "policy": "visible_salient_objects_attributes_directed_relations_source_first_human_identity_v1",
        }
        write(self.output / "workspace.json", self.identity)

    def unchanged(self):
        if sha256(self.manifest_path) != self.identity["source_manifest_sha256"]:
            raise ValueError("Manifest gốc đã thay đổi; cần workspace version mới.")

    def directory(self, identifier):
        if identifier not in self.records:
            raise ValueError("Không có cặp train/val này; test chưa được mở.")
        return self.output / "events" / digest(identifier)

    def history(self, identifier):
        parent, revision, rows = None, 0, []
        for path in sorted(self.directory(identifier).glob("*.json")):
            event = read(path)
            checksum = digest({k: v for k, v in event.items() if k != "event_sha256"})
            if (
                event["event_sha256"] != checksum
                or path.name != f"{event['revision']:06d}-{checksum}.json"
                or event["parent_sha256"] != parent
                or event["revision"] != revision + 1
                or event["id"] != identifier
                or event["workspace_sha256"] != digest(self.identity)
            ):
                raise ValueError("Lịch sử annotation thiếu hoặc bị thay đổi.")
            rows.append(event)
            parent, revision = checksum, event["revision"]
        return rows

    def latest(self, identifier):
        events = self.history(identifier)
        return (
            events[-1]
            if events
            else {
                "id": identifier,
                "revision": 0,
                "event_sha256": None,
                "source_locked": False,
                "source_graph": copy.deepcopy(EMPTY),
                "target_graph": None,
                "source_points": {},
                "target_points": {},
                "source_lock_sha256": None,
                "status": "draft",
                "annotator": "",
                "notes": "",
                "reason": "",
                "instruction_matches": False,
                "unchanged_checked": False,
            }
        )

    def bootstrap(self):
        self.unchanged()
        rows = []
        review = set(self.review_ids)
        ordered = sorted(self.records, key=lambda i: (i not in review, digest([42, i])))
        for identifier in ordered:
            row, state = self.records[identifier], self.latest(identifier)
            rows.append(
                {
                    "id": identifier,
                    "split": row["split"],
                    "status": state["status"],
                    "pilot": identifier in review,
                    "source_group_id": row["source_group_id"],
                }
            )
        return {
            "version": VERSION,
            "workspace_sha256": digest(self.identity),
            "cases": rows,
            "review_ids": self.review_ids,
            "counts": dict(Counter(r["status"] for r in rows)),
            "training_ready": False,
        }

    def case(self, identifier):
        self.unchanged()
        state = self.latest(identifier)
        row = self.records[identifier]
        source = {k: row["source_image"][k] for k in ("width", "height", "sha256")}
        result = {
            "id": identifier,
            "split": row["split"],
            "state": state,
            "source_image": source,
            "instruction": None,
            "target_image": None,
        }
        if state["source_locked"]:
            result["instruction"] = row["instruction"]
            result["target_image"] = {k: row["target_image"][k] for k in ("width", "height", "sha256")}
        result["validation"] = (
            validate_pair(state["source_graph"], state["target_graph"]) if state["target_graph"] else None
        )
        return result

    def image(self, identifier, role):
        self.directory(identifier)
        if role not in ("source", "target") or (role == "target" and not self.latest(identifier)["source_locked"]):
            raise ValueError("Hãy xác nhận graph nguồn trước khi xem ảnh đích.")
        self.unchanged()
        meta = self.records[identifier][role + "_image"]
        path = (self.root / meta["path"]).resolve()
        if not path.is_relative_to(self.root / "data"):
            raise ValueError("Ảnh ngoài thư mục dữ liệu.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != meta["sha256"]:
            raise ValueError("Ảnh không khớp hash trong manifest.")
        return content, path.suffix.lower()

    @contextmanager
    def locked(self):
        with (self.output / ".write.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def save(self, identifier, action, payload):
        self.directory(identifier)
        self.unchanged()
        if action not in ("draft", "lock_source", "reopen_source", "complete", "defer"):
            raise ValueError("Thao tác lưu không hợp lệ.")
        with self.locked():
            current = self.latest(identifier)
            if type(payload.get("base_revision")) is not int or payload["base_revision"] != current["revision"]:
                raise Conflict(
                    "Có bản mới từ tab/người khác. Bản đang nhập vẫn được giữ trong trình duyệt; tải lại hoặc tải bản nháp để đối chiếu."
                )
            state = copy.deepcopy(current)
            for key in ("annotator", "notes", "reason"):
                value = payload.get(key, state[key])
                if not isinstance(value, str) or len(value) > (120 if key == "annotator" else 4000):
                    raise ValueError("Tên người gán/ghi chú không hợp lệ.")
                state[key] = value.strip()
            if action == "reopen_source":
                state.update(
                    source_locked=False,
                    source_lock_sha256=None,
                    target_graph=None,
                    target_points={},
                    instruction_matches=False,
                    unchanged_checked=False,
                    status="draft",
                )
            else:
                source = graph(payload.get("source_graph", state["source_graph"]))
                source_points = points(payload.get("source_points", state["source_points"]), source)
                if current["source_locked"] and (
                    source != current["source_graph"] or source_points != current["source_points"]
                ):
                    raise ValueError("Graph nguồn đã khóa. Dùng Sửa lại nguồn để tạo lượt gán mới.")
                state["source_graph"], state["source_points"] = source, source_points
                if action == "lock_source":
                    if current["source_locked"] or not state["annotator"]:
                        raise ValueError("Nhập tên người gán và xác nhận nguồn đúng một lần.")
                    parsed = parse_graph(source)
                    if not parsed.nodes or len(parsed.nodes) > 20 or len(parsed.edges) > 60:
                        raise ValueError("Nguồn cần 1–20 đối tượng, tối đa 60 quan hệ và không thiếu nhãn.")
                    if any(not re.fullmatch(r"n[0-9]{3,}", n.id) for n in parsed.nodes) or any(
                        not re.fullmatch(r"e[0-9]{3,}", e.id) for e in parsed.edges
                    ):
                        raise ValueError("ID nguồn không hợp lệ.")
                    self.image(identifier, "source")
                    normalized, nodes, _ = canonical_source(parsed)
                    state["source_graph"] = json.loads(json.dumps(normalized.to_dict()))
                    state["source_points"] = {nodes[k]: v for k, v in source_points.items()}
                    state["target_graph"] = copy.deepcopy(state["source_graph"])
                    state["target_points"] = {}
                    state["source_locked"] = True
                    state["source_lock_sha256"] = digest(
                        {
                            "source_graph": state["source_graph"],
                            "source_points": state["source_points"],
                            "image_sha256": self.records[identifier]["source_image"]["sha256"],
                            "annotator": state["annotator"],
                            "next_revision": current["revision"] + 1,
                        }
                    )
                elif current["source_locked"]:
                    state["target_graph"] = graph(payload.get("target_graph", state["target_graph"]))
                    state["target_points"] = points(
                        payload.get("target_points", state["target_points"]), state["target_graph"]
                    )
                elif payload.get("target_graph") is not None:
                    raise ValueError("Chưa xác nhận nguồn; chưa được gán graph đích.")
                for key in ("instruction_matches", "unchanged_checked"):
                    value = payload.get(key, state[key])
                    if type(value) is not bool:
                        raise ValueError("Cờ kiểm tra phải là boolean.")
                    state[key] = value
                state["status"] = "source_locked" if state["source_locked"] else "draft"
                if action == "defer":
                    if not state["annotator"] or not state["reason"]:
                        raise ValueError("Ghi tên người gán và lý do tạm bỏ qua.")
                    state["status"] = "deferred"
                if action == "complete":
                    if (
                        not state["source_locked"]
                        or not state["annotator"]
                        or not all(state[k] for k in ("instruction_matches", "unchanged_checked"))
                    ):
                        raise ValueError("Cần tên người gán, graph nguồn đã xác nhận và đủ hai mục đối chiếu ảnh.")
                    validation = validate_pair(state["source_graph"], state["target_graph"])
                    if not validation["valid"]:
                        raise ValueError("; ".join(validation["problems"]))
                    self.image(identifier, "source")
                    self.image(identifier, "target")
                    state["status"] = "completed"
            # An exact autosave is a no-op; don't invalidate a completed label.
            mutable = (
                "source_graph",
                "target_graph",
                "source_points",
                "target_points",
                "annotator",
                "notes",
                "reason",
                "instruction_matches",
                "unchanged_checked",
            )
            if action == "draft" and all(state[k] == current[k] for k in mutable):
                return self.case(identifier)
            event = {
                **{
                    k: v
                    for k, v in state.items()
                    if k
                    not in (
                        "event_sha256",
                        "revision",
                        "parent_sha256",
                        "action",
                        "workspace_sha256",
                        "created_at",
                        "code",
                    )
                },
                "revision": current["revision"] + 1,
                "parent_sha256": current["event_sha256"],
                "workspace_sha256": digest(self.identity),
                "action": action,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code": code_hashes(),
            }
            event["event_sha256"] = digest(event)
            write(self.directory(identifier) / f"{event['revision']:06d}-{event['event_sha256']}.json", event)
            return self.case(identifier)

    def compile(self):
        """Export a snapshot for loader/review; never claim training readiness."""
        from tools.edit_model_v1.data import load_examples

        self.unchanged()
        with self.locked():
            states = {identifier: self.latest(identifier) for identifier in self.records}
        records, counts = [], Counter()
        for identifier, state in states.items():
            counts[state["status"]] += 1
            if state["status"] != "completed":
                continue
            validation = validate_pair(state["source_graph"], state["target_graph"])
            if not validation["valid"] or not state["source_lock_sha256"] or not state["annotator"]:
                raise ValueError("Annotation hoàn thành không còn hợp lệ.")
            self.image(identifier, "source")
            self.image(identifier, "target")
            row = self.records[identifier]
            records.append(
                {
                    **{k: row[k] for k in ("id", "split", "source_group_id", "instruction")},
                    "source_graph": validation["source_graph"],
                    "target_graph": validation["target_graph"],
                    "operations": validation["operations"],
                    "status": "candidate",
                    "eligible_for_training": True,
                    "problems": [],
                    "human_review_status": "pending",
                    "label_origin": "human_image_annotation",
                    "annotator": state["annotator"],
                    "annotation_event_sha256": state["event_sha256"],
                    "source_lock_sha256": state["source_lock_sha256"],
                }
            )
        manifest = {
            **self.original,
            "records": list(self.records.values()),
            "splits": {s: [i for i, r in self.records.items() if r["split"] == s] for s in ("train", "val", "test")},
            "manual_annotation_version": VERSION,
            "original_manifest_sha256": self.identity["source_manifest_sha256"],
            "graph_labels_generated": bool(records),
            "graph_edit_requirement_complete": False,
        }
        selection = {i: s["event_sha256"] for i, s in states.items()}
        snapshot = digest({"workspace": self.identity, "events": selection})
        directory = self.output / "exports" / snapshot
        write(directory / "manifest.json", manifest)
        edits = {
            "dataset_manifest_sha256": sha256(directory / "manifest.json"),
            "label_origin": "human_image_annotation",
            "records": records,
        }
        write(directory / "edits.json", edits)
        _, loader = load_examples(directory / "manifest.json", directory / "edits.json")
        loader["label_origin"] = "human_image_annotation"
        report = {
            "snapshot": snapshot,
            "directory": str(directory),
            "status_counts": dict(counts),
            "completed": len(records),
            "loader": loader,
            "review_ids": self.review_ids,
            "independent_review_completed": False,
            "ready_for_training": False,
            "minimum300_count_reached": len(records) >= 300,
            "events": selection,
            "source_manifest_sha256": self.identity["source_manifest_sha256"],
        }
        write(directory / "report.json", report)
        return {"manifest": manifest, "edits": edits, "report": report}
