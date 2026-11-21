"""Versioned pair-visible workspace; v1 snapshots and all old files are preserved."""

import copy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from common.io import digest, fixed, read, sha256, write

from task2_graph_editing.magicbrush.annotation.store import EMPTY, MANIFEST
from task2_graph_editing.magicbrush.annotation.store import OUTPUT as LEGACY_OUTPUT
from task2_graph_editing.magicbrush.annotation.store import ROOT, Conflict
from task2_graph_editing.magicbrush.annotation.store import Store as V1Store
from task2_graph_editing.magicbrush.annotation.store import graph, points, validate_pair
from task2_graph_editing.magicbrush.pair_generation import VERSION
from task2_graph_editing.magicbrush.pair_generation.proposal import parse

OUTPUT = ROOT / "data/processed/level2/magicbrush-manual-v2"


def code_hashes():
    files = list(Path(__file__).parent.glob("*.py")) + list(Path(__file__).with_name("assets").glob("*"))
    files += [
        ROOT / "task2_graph_editing/magicbrush/annotation/store.py",
        ROOT / "common/editing/operations.py",
        ROOT / "task2_graph_editing/magicbrush/sequence_model/codec.py",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


class Store(V1Store):
    def __init__(self, manifest=MANIFEST, output=OUTPUT, *, legacy_output=LEGACY_OUTPUT, root=ROOT):
        if Path(output).resolve() == Path(legacy_output).resolve():
            raise ValueError("V2 cần thư mục riêng; không ghi đè workspace v1.")
        legacy = V1Store(manifest, legacy_output, root=root)
        self.__dict__.update(legacy.__dict__)
        self.output = Path(output).resolve()
        self.identity = {
            **legacy.identity,
            "schema": VERSION,
            "policy": "both_images_instruction_visible_model_assisted_proposals_v2",
            "source_identity": "source_only_canonicalization",
            "legacy_workspace_sha256": digest(legacy.identity),
        }
        write(self.output / "workspace.json", self.identity)
        receipt_path = self.output / "legacy-snapshot.json"
        if not receipt_path.exists():
            files, states = {}, {}
            with legacy.locked():
                for identifier in self.records:
                    history = legacy.history(identifier)
                    if not history:
                        continue
                    states[identifier] = history[-1]
                    for path in sorted(legacy.directory(identifier).glob("*.json")):
                        relative = Path("legacy-v1") / path.relative_to(legacy.output)
                        fixed(self.output / relative, path.read_bytes())
                        files[str(relative)] = sha256(path)
            write(receipt_path, {"workspace_sha256": digest(legacy.identity), "files": files, "states": states})
        self.legacy_snapshot = read(receipt_path)
        if self.legacy_snapshot["workspace_sha256"] != digest(legacy.identity):
            raise ValueError("Snapshot v1 không khớp provenance.")
        for relative, checksum in self.legacy_snapshot["files"].items():
            path = (self.output / relative).resolve()
            if not path.is_relative_to(self.output / "legacy-v1") or sha256(path) != checksum:
                raise ValueError("Lịch sử v1 đã sao lưu bị thay đổi.")
        for identifier, event in self.legacy_snapshot["states"].items():
            path = (
                self.output
                / "legacy-v1"
                / "events"
                / digest(identifier)
                / f"{event['revision']:06d}-{event['event_sha256']}.json"
            )
            if read(path) != event:
                raise ValueError("Bản nhãn v1 không khớp lịch sử đã sao lưu.")

    def latest(self, identifier):
        events = self.history(identifier)
        if events:
            return events[-1]
        old = copy.deepcopy(self.legacy_snapshot["states"].get(identifier))
        if old is None:
            old = V1Store.latest(self, identifier)
        else:
            old.update(
                legacy_event_sha256=old["event_sha256"],
                legacy_revision=old["revision"],
                annotation_origin="legacy_v1_human_annotation",
            )
            if old["status"] == "source_locked":
                old.update(legacy_status="source_locked", status="draft")
        old.update(revision=0, event_sha256=None, source_locked=False)
        old["target_graph"] = old["target_graph"] if old["target_graph"] is not None else copy.deepcopy(EMPTY)
        old.setdefault("annotation_origin", "manual_pair_annotation")
        return old

    def bootstrap(self):
        result = super().bootstrap()
        result.update(version=VERSION, both_images_visible=True, migrated_cases=len(self.legacy_snapshot["states"]))
        return result

    def case(self, identifier):
        self.unchanged()
        self.directory(identifier)
        state, row = self.latest(identifier), self.records[identifier]
        return {
            "id": identifier,
            "split": row["split"],
            "state": state,
            "instruction": row["instruction"],
            **{
                role + "_image": {k: row[role + "_image"][k] for k in ("width", "height", "sha256")}
                for role in ("source", "target")
            },
            "validation": validate_pair(state["source_graph"], state["target_graph"]),
        }

    def image(self, identifier, role):
        # Reuse v1 allowlist/hash checks without its source-first gate. No data is
        # changed and test IDs are still rejected by directory().
        import hashlib

        self.directory(identifier)
        self.unchanged()
        if role not in ("source", "target"):
            raise ValueError("Vai trò ảnh không hợp lệ.")
        meta = self.records[identifier][role + "_image"]
        path = (self.root / meta["path"]).resolve()
        if not path.is_relative_to(self.root / "data"):
            raise ValueError("Ảnh ngoài thư mục dữ liệu.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != meta["sha256"]:
            raise ValueError("Ảnh không khớp hash trong manifest.")
        return content, path.suffix.lower()

    def append(self, identifier, action, current, state):
        event = {
            **{
                k: v
                for k, v in state.items()
                if k
                not in ("event_sha256", "revision", "parent_sha256", "action", "workspace_sha256", "created_at", "code")
            },
            "revision": current["revision"] + 1,
            "parent_sha256": current["event_sha256"],
            "workspace_sha256": digest(self.identity),
            "action": action,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code": code_hashes(),
            "annotation_context": "both_images_and_instruction_visible",
        }
        event["event_sha256"] = digest(event)
        write(self.directory(identifier) / f"{event['revision']:06d}-{event['event_sha256']}.json", event)
        return self.case(identifier)

    def import_json(self, identifier, text, base_revision, annotator=""):
        self.unchanged()
        self.directory(identifier)
        proposal = parse(text, identifier)  # Validate completely before any write.
        if not isinstance(annotator, str) or len(annotator) > 120:
            raise ValueError("Tên người gán không hợp lệ.")
        with self.locked():
            current = self.latest(identifier)
            if type(base_revision) is not int or base_revision != current["revision"]:
                raise Conflict("Có bản lưu mới hơn. JSON chưa được áp dụng; hãy tải lại để đối chiếu.")
            combined_notes = "\n".join(dict.fromkeys(s for s in (current["notes"], proposal["notes"]) if s))
            if len(combined_notes) > 4000:
                raise ValueError(
                    "Ghi chú cũ và JSON vượt 4000 ký tự; rút gọn ghi chú rồi nhập lại. Chưa thay đổi dữ liệu."
                )
            state = {
                **copy.deepcopy(current),
                "source_graph": proposal["source_graph"],
                "target_graph": proposal["target_graph"],
                "source_points": {},
                "target_points": {},
                "status": "draft",
                "source_locked": False,
                "instruction_matches": False,
                "unchanged_checked": False,
                "source_lock_sha256": None,
                "annotation_origin": "external_json_proposal",
                "annotator": annotator.strip() or current["annotator"],
                "proposal": {
                    "raw_json": text,
                    "sha256": proposal["proposal_sha256"],
                    "origin": "user_supplied_external_model_or_manual_json",
                    "human_review_status": "pending",
                },
                "notes": combined_notes,
            }
            return self.append(identifier, "import_json_proposal", current, state)

    def save(self, identifier, action, payload):
        self.directory(identifier)
        self.unchanged()
        if action not in ("draft", "complete", "defer", "copy_source"):
            raise ValueError("V2 không yêu cầu xác nhận nguồn. Có thể sửa cả hai graph.")
        with self.locked():
            current = self.latest(identifier)
            if type(payload.get("base_revision")) is not int or payload["base_revision"] != current["revision"]:
                raise Conflict("Có bản mới từ tab/người khác. Bản đang nhập vẫn được giữ trong trình duyệt.")
            state = copy.deepcopy(current)
            for key in ("annotator", "notes", "reason"):
                value = payload.get(key, state[key])
                if not isinstance(value, str) or len(value) > (120 if key == "annotator" else 4000):
                    raise ValueError("Tên hoặc ghi chú không hợp lệ.")
                state[key] = value.strip()
            for role in ("source", "target"):
                state[role + "_graph"] = graph(payload.get(role + "_graph", state[role + "_graph"]))
                state[role + "_points"] = points(
                    payload.get(role + "_points", state[role + "_points"]), state[role + "_graph"]
                )
            for key in ("instruction_matches", "unchanged_checked"):
                state[key] = payload.get(key, state[key])
                if type(state[key]) is not bool:
                    raise ValueError("Cờ đối chiếu phải là boolean.")
            if action == "draft" and all(state[k] == current[k] for k in state if k not in ("status",)):
                return self.case(identifier)
            state.update(status="draft", source_locked=False)
            if action == "copy_source":
                state.update(
                    target_graph=copy.deepcopy(state["source_graph"]),
                    target_points={},
                    instruction_matches=False,
                    unchanged_checked=False,
                )
            if action == "defer":
                if not state["annotator"] or not state["reason"]:
                    raise ValueError("Điền tên và lý do tạm bỏ qua.")
                state["status"] = "deferred"
            if action == "complete":
                if not state["annotator"] or not all(state[k] for k in ("instruction_matches", "unchanged_checked")):
                    raise ValueError("Điền tên và đối chiếu đủ hai mục trước khi hoàn thành.")
                validation = validate_pair(state["source_graph"], state["target_graph"])
                if not validation["valid"]:
                    raise ValueError("; ".join(validation["problems"]))
                for role in ("source", "target"):
                    self.image(identifier, role)
                state["status"] = "completed"
                state["source_validation_sha256"] = digest(state["source_graph"])
                state["proposal_checked_by_annotator"] = bool(state.get("proposal"))
            return self.append(identifier, action, current, state)

    def compile(self):
        from task2_graph_editing.magicbrush.sequence_model.data import load_examples

        self.unchanged()
        with self.locked():
            states = {i: self.latest(i) for i in self.records}
        records = []
        for identifier, state in states.items():
            if state["status"] != "completed":
                continue
            validation = validate_pair(state["source_graph"], state["target_graph"])
            if not validation["valid"]:
                raise ValueError("Graph hoàn thành không hợp lệ.")
            for role in ("source", "target"):
                self.image(identifier, role)
            records.append(
                {
                    **{k: self.records[identifier][k] for k in ("id", "split", "source_group_id", "instruction")},
                    **{k: validation[k] for k in ("source_graph", "target_graph", "operations")},
                    "status": "candidate",
                    "eligible_for_training": True,
                    "problems": [],
                    "human_review_status": "pending",
                    "label_origin": state["annotation_origin"],
                    "annotator": state["annotator"],
                    "annotation_event_sha256": state["event_sha256"] or state.get("legacy_event_sha256"),
                    "annotation_context": state.get("annotation_context", "legacy_v1_source_first"),
                    "proposal_sha256": state.get("proposal", {}).get("sha256"),
                }
            )
        manifest = {
            **self.original,
            "records": list(self.records.values()),
            "splits": {s: [i for i, r in self.records.items() if r["split"] == s] for s in ("train", "val", "test")},
            "manual_annotation_version": VERSION,
            "original_manifest_sha256": self.identity["source_manifest_sha256"],
            "graph_edit_requirement_complete": False,
        }
        snapshot = digest(
            [
                self.identity,
                sha256(self.output / "legacy-snapshot.json"),
                {i: s["event_sha256"] for i, s in states.items()},
            ]
        )
        directory = self.output / "exports" / snapshot
        write(directory / "manifest.json", manifest)
        edits = {
            "dataset_manifest_sha256": sha256(directory / "manifest.json"),
            "label_origin": "paired_context_human_or_model_assisted_annotation",
            "records": records,
        }
        write(directory / "edits.json", edits)
        _, loader = load_examples(directory / "manifest.json", directory / "edits.json")
        loader["label_origin"] = edits["label_origin"]
        report = {
            "snapshot": snapshot,
            "directory": str(directory),
            "completed": len(records),
            "status_counts": dict(Counter(s["status"] for s in states.values())),
            "loader": loader,
            "ready_for_training": False,
            "independent_review_completed": False,
            "source_manifest_sha256": self.identity["source_manifest_sha256"],
        }
        write(directory / "report.json", report)
        return {"manifest": manifest, "edits": edits, "report": report}
