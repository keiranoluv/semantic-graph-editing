"""TransE with independent KG namespaces and provenance-preserving loading."""

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class KG:
    entities: list[str]
    relations: list[str]
    triples: np.ndarray
    source: str
    hashes: dict[str, str]
    metadata: dict


def load_kg(folder: Path) -> KG:
    """Validate B.1/B.2.1 schemas and deduplicate evidence-only CN rows."""
    names = ("entities.json", "relations.json", "triples.tsv", "metadata.json")
    paths = {name: folder / name for name in names}
    docs = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in names if name.endswith(".json")}
    metadata = docs["metadata.json"]
    source = metadata.get("source")
    if metadata.get("schema_version") != 1 or source not in ("wordnet", "conceptnet"):
        raise ValueError("Unsupported KG metadata schema/source")
    prefix = "wn:" if source == "wordnet" else "cn:"
    entities, relations = docs["entities.json"].get("entities"), docs["relations.json"].get("relations")
    if (
        not isinstance(entities, list)
        or not entities
        or any(not isinstance(x, str) or not x.startswith(prefix) for x in entities)
        or len(set(entities)) != len(entities)
    ):
        raise ValueError("Invalid/duplicate entity IDs or namespace")
    if (
        not isinstance(relations, list)
        or not relations
        or any(not isinstance(x, str) or x == "SameAs" for x in relations)
        or len(set(relations)) != len(relations)
    ):
        raise ValueError("Invalid relations; SameAs is explicitly forbidden in C.1")
    rows = set()
    with paths["triples.tsv"].open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"subject_id", "relation_id", "object_id"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError("Invalid triple TSV columns")
        for item in reader:
            triple = tuple(int(item[k]) for k in ("subject_id", "relation_id", "object_id"))
            if not (
                0 <= triple[0] < len(entities) and 0 <= triple[1] < len(relations) and 0 <= triple[2] < len(entities)
            ):
                raise ValueError("Triple ID outside vocabulary")
            rows.add(triple)
    if not rows:
        raise ValueError("Empty KG")
    if source == "wordnet" and metadata.get("triple_count") != len(rows):
        raise ValueError("WordNet triple count mismatch")
    if source == "conceptnet" and metadata.get("unique_triple_count") != len(rows):
        raise ValueError("ConceptNet unique-triple count mismatch")
    return KG(
        entities,
        relations,
        np.asarray(sorted(rows), dtype=np.int64),
        source,
        {name: sha256(path) for name, path in paths.items()},
        metadata,
    )


def check_gate(mode: str, kg: KG, folder: Path, review_manifest: Path | None = None) -> None:
    if mode == "draft":
        if review_manifest is not None:
            raise ValueError("Draft mode does not take a review manifest")
        return
    if mode != "reviewed":
        raise ValueError("Unknown training mode")
    if review_manifest is None:
        raise ValueError("C-Reviewed requires an externally human-reviewed manifest")
    m = json.loads(review_manifest.read_text(encoding="utf-8"))
    if m.get("schema_version") != 1 or m.get("kg_source") != kg.source:
        raise ValueError("Review manifest missing human approval or incorrect source")
    if m.get("kg_sha256") != kg.hashes or m.get("kg_path") != str(folder.resolve()):
        raise ValueError("Review manifest does not bind all exact input files/path")
    if kg.metadata.get("ready_for_gat_kg_training") is not True:
        raise ValueError("Source KG metadata explicitly not training-ready")
    if m.get("review_status") == "human_approved_batch":
        _check_batch_review(m, kg, folder)
        return
    if m.get("review_status") != "human_approved":
        raise ValueError("Review manifest missing human approval")
    if not isinstance(m.get("reviews"), list) or len(m["reviews"]) != 150:
        raise ValueError("Require 150 individually reviewed VG150 class decisions")
    ids = set()
    for item in m["reviews"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid review entry")
        i = item.get("class_id")
        if type(i) is not int or i not in range(1, 151) or i in ids:
            raise ValueError("Review class IDs must be unique and complete")
        ids.add(i)
        if item.get("decision") not in ("accept", "reject", "defer"):
            raise ValueError("Missing review decision")
        if any(
            not isinstance(item.get(k), str) or not item[k].strip()
            for k in ("reviewer", "evidence_reference", "rationale")
        ):
            raise ValueError("Review requires named human, evidence and rationale")
        if item["reviewer"].strip().casefold() in {"ai", "openai", "chatgpt", "assistant", "llm"}:
            raise ValueError("AI reviewer is not human review evidence")
    batch_approved_ai = (
        kg.source == "conceptnet"
        and m.get("review_basis") == "human_batch_approved_ai_proposal"
        and kg.metadata.get("human_batch_approved_class_count") == 150
    )
    if kg.metadata.get("curation_tier", "").upper().find("PROVISIONAL") >= 0 or (
        kg.metadata.get("human_verified_class_count") == 0 and not batch_approved_ai
    ):
        raise ValueError("Provisional/unverified KG may not run in C-Reviewed")
    if kg.source == "conceptnet":
        _check_conceptnet_review(m, kg, folder)


def _check_conceptnet_review(m: dict, kg: KG, folder: Path) -> None:
    """Bind individual URI decisions and source evidence to the exact graph."""
    basis = m.get("review_basis")
    if basis not in {
        "individual_conceptnet_class_uri_selection",
        "human_batch_approved_ai_proposal",
        "human_mixed_direct_and_ai_batch",
    }:
        raise ValueError("Unsupported ConceptNet review basis")
    if basis != kg.metadata.get("review_basis"):
        raise ValueError("ConceptNet KG/review basis mismatch")
    for name, key in (
        ("review.csv", "review_csv_sha256"),
        ("class_concept_candidates.json", "class_concept_candidates_sha256"),
        ("provenance.json", "provenance_sha256"),
        ("class_entity_ids.json", "class_entity_ids_sha256"),
        ("class_concept_links.json", "class_concept_links_sha256"),
    ):
        if sha256(folder / name) != m.get(key):
            raise ValueError(f"ConceptNet review artifact hash mismatch: {name}")
    if sha256(folder / "review.csv") != kg.metadata.get("input_review_sha256"):
        raise ValueError("ConceptNet KG metadata/review CSV mismatch")
    ids = json.loads((folder / "class_entity_ids.json").read_text(encoding="utf-8"))["class_entity_ids"]
    links = json.loads((folder / "class_concept_links.json").read_text(encoding="utf-8"))["classes"]
    candidates = json.loads((folder / "class_concept_candidates.json").read_text(encoding="utf-8"))["classes"]
    with (folder / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not all(len(x) == 151 for x in (ids, links, candidates)) or len(rows) != 150:
        raise ValueError("ConceptNet review class count mismatch")
    if ids[0] is not None or links[0].get("decision") != "background":
        raise ValueError("ConceptNet background class must be unlinked")
    evidence = set()
    with (folder / "triples.tsv").open(encoding="utf-8", newline="") as stream:
        for edge in csv.DictReader(stream, delimiter="\t"):
            for endpoint in ("subject_id", "object_id"):
                evidence.add((kg.entities[int(edge[endpoint])], edge["assertion_uri"]))
    accepted = 0
    for i, (item, row) in enumerate(zip(m["reviews"], rows, strict=True), start=1):
        uri = item.get("selected_uri")
        decision = item["decision"]
        if (
            item.get("class_id") != i
            or row.get("class_id") != str(i)
            or row.get("label") != item.get("label")
            or row.get("conceptnet_uri_candidates") != ";".join(candidates[i].get("concept_uris", []))
            or candidates[i].get("label") != item.get("label")
            or links[i].get("label") != item.get("label")
            or links[i].get("decision") != decision
            or links[i].get("selected_uri") != uri
            or row.get("decision", "").strip().lower() != decision
            or row.get("selected_uri", "").strip() != (uri or "")
            or row.get("reviewer", "").strip() != item.get("reviewer")
            or row.get("evidence_reference", "").strip() != item.get("evidence_reference")
            or row.get("rationale", "").strip() != item.get("rationale")
        ):
            raise ValueError(f"ConceptNet review row mismatch at class {i}")
        if decision == "accept":
            if (
                uri not in candidates[i].get("concept_uris", [])
                or type(ids[i]) is not int
                or not 0 <= ids[i] < len(kg.entities)
                or kg.entities[ids[i]] != f"cn:{uri}"
                or (f"cn:{uri}", item["evidence_reference"]) not in evidence
            ):
                raise ValueError(f"ConceptNet URI/entity/evidence mismatch at class {i}")
            accepted += 1
        elif uri is not None or ids[i] is not None:
            raise ValueError(f"Rejected/deferred ConceptNet class must be unlinked: {i}")
    reviewed_count = (
        kg.metadata.get("human_verified_class_count", 0) + kg.metadata.get("human_batch_approved_class_count", 0)
        if basis == "human_mixed_direct_and_ai_batch"
        else kg.metadata.get("human_batch_approved_class_count")
        if basis == "human_batch_approved_ai_proposal"
        else kg.metadata.get("human_verified_class_count")
    )
    if accepted != kg.metadata.get("resolved_class_count") or reviewed_count != 150:
        raise ValueError("ConceptNet reviewed KG class coverage mismatch")
    if basis == "human_batch_approved_ai_proposal":
        _check_conceptnet_batch_approval(m, folder, rows)
    if basis == "human_mixed_direct_and_ai_batch":
        _check_conceptnet_mixed_approval(m, kg, folder, rows)


def _check_conceptnet_mixed_approval(m: dict, kg: KG, folder: Path, rows: list[dict]) -> None:
    """Verify 134 approved AI rows and 16 direct human decisions separately."""
    if (
        m.get("human_direct_class_count") != 16
        or m.get("human_batch_approved_class_count") != 134
        or kg.metadata.get("human_verified_class_count") != 16
        or kg.metadata.get("human_batch_approved_class_count") != 134
    ):
        raise ValueError("ConceptNet mixed review count mismatch")
    for name, key in (
        ("approval.json", "approval_sha256"),
        ("selection.csv", "selection_sha256"),
        ("direct_decisions.csv", "direct_decisions_sha256"),
        ("review_suggestions.ai-draft.csv", "proposal_sha256"),
        ("review_focus.ai-draft.csv", "focus_sha256"),
    ):
        if sha256(folder / name) != m.get(key):
            raise ValueError(f"ConceptNet mixed approval artifact hash mismatch: {name}")
    approval = json.loads((folder / "approval.json").read_text(encoding="utf-8"))
    accepted_ids = approval.get("accepted_focus_class_ids")
    if (
        approval.get("schema_version") != 1
        or approval.get("approval_scope") != "conceptnet_ai_134_plus_16_direct_focus"
        or approval.get("approver") != m.get("approved_by")
        or approval.get("proposal_sha256") != m["proposal_sha256"]
        or approval.get("focus_sha256") != m["focus_sha256"]
        or approval.get("selection_sha256") != m["selection_sha256"]
        or approval.get("approve_non_focus_ai_proposals") is not True
        or approval.get("reject_remaining_focus") is not True
        or not isinstance(approval.get("statement"), str)
        or len(approval["statement"].strip()) < 20
        or not isinstance(accepted_ids, list)
        or any(type(i) is not int for i in accepted_ids)
        or accepted_ids != sorted(set(accepted_ids))
    ):
        raise ValueError("ConceptNet mixed approval declaration mismatch")
    with (folder / "review_suggestions.ai-draft.csv").open(encoding="utf-8", newline="") as stream:
        proposals = list(csv.DictReader(stream))
    with (folder / "review_focus.ai-draft.csv").open(encoding="utf-8", newline="") as stream:
        focus = list(csv.DictReader(stream))
    with (folder / "selection.csv").open(encoding="utf-8", newline="") as stream:
        selections = list(csv.DictReader(stream))
    with (folder / "direct_decisions.csv").open(encoding="utf-8", newline="") as stream:
        direct = list(csv.DictReader(stream))
    focus_ids = [int(item["class_id"]) for item in focus]
    if (
        len(proposals) != 150
        or len(focus_ids) != 16
        or focus_ids != sorted(set(focus_ids))
        or {int(item["class_id"]) for item in proposals if item["proposed_decision"] != "accept"} != set(focus_ids)
        or [int(item["class_id"]) for item in selections] != accepted_ids
        or [int(item["class_id"]) for item in direct] != focus_ids
        or not set(accepted_ids).issubset(focus_ids)
    ):
        raise ValueError("ConceptNet mixed approval focus/selection mismatch")
    direct_by_id = {int(item["class_id"]): item for item in direct}
    selection_by_id = {int(item["class_id"]): item for item in selections}
    for i, (row, proposal) in enumerate(zip(rows, proposals, strict=True), start=1):
        if (
            proposal.get("class_id") != str(i)
            or proposal.get("label") != row["label"]
            or proposal.get("conceptnet_uri_candidates") != row["conceptnet_uri_candidates"]
            or row["reviewer"] != m["approved_by"]
        ):
            raise ValueError(f"ConceptNet mixed proposal/class mismatch at {i}")
        if i in direct_by_id:
            item = direct_by_id[i]
            if any(
                item.get(key) != row.get(key)
                for key in ("label", "decision", "selected_uri", "reviewer", "evidence_reference", "rationale")
            ):
                raise ValueError(f"ConceptNet direct decision mismatch at {i}")
            if i in selection_by_id:
                selected = selection_by_id[i]
                if row["decision"] != "accept" or any(
                    selected.get(key) != row.get(key) for key in ("selected_uri", "evidence_reference", "rationale")
                ):
                    raise ValueError(f"ConceptNet accepted focus selection mismatch at {i}")
            elif row["decision"] != "reject" or row["selected_uri"]:
                raise ValueError(f"ConceptNet rejected focus decision mismatch at {i}")
        elif (
            proposal.get("proposed_decision") != "accept"
            or row["decision"] != "accept"
            or proposal.get("proposed_uri") != row["selected_uri"]
            or proposal.get("evidence_reference") != row["evidence_reference"]
            or not row["rationale"].startswith(
                f"Batch-approved AI proposal {m['proposal_sha256'][:12]} by {m['approved_by']}; "
            )
        ):
            raise ValueError(f"ConceptNet approved AI proposal mismatch at {i}")


def _check_conceptnet_batch_approval(m: dict, folder: Path, rows: list[dict]) -> None:
    for name, key in (
        ("approval.json", "approval_sha256"),
        ("review_suggestions.ai-draft.csv", "proposal_sha256"),
        ("review_focus.ai-draft.csv", "focus_sha256"),
    ):
        if sha256(folder / name) != m.get(key):
            raise ValueError(f"ConceptNet batch approval artifact hash mismatch: {name}")
    approval = json.loads((folder / "approval.json").read_text(encoding="utf-8"))
    if (
        approval.get("schema_version") != 1
        or approval.get("approval_scope") != "all_150_conceptnet_ai_proposals"
        or approval.get("approver") != m.get("approved_by")
        or approval.get("proposal_sha256") != m["proposal_sha256"]
        or approval.get("focus_sha256") != m["focus_sha256"]
        or not isinstance(approval.get("statement"), str)
        or len(approval["statement"].strip()) < 20
    ):
        raise ValueError("ConceptNet batch approval declaration mismatch")
    with (folder / "review_suggestions.ai-draft.csv").open(encoding="utf-8", newline="") as stream:
        proposals = list(csv.DictReader(stream))
    if len(proposals) != 150:
        raise ValueError("ConceptNet batch approval needs all 150 AI proposals")
    for i, (row, proposal) in enumerate(zip(rows, proposals, strict=True), start=1):
        if (
            proposal.get("class_id") != str(i)
            or proposal.get("label") != row["label"]
            or proposal.get("conceptnet_uri_candidates") != row["conceptnet_uri_candidates"]
            or proposal.get("proposed_decision") != row["decision"]
            or proposal.get("proposed_uri", "") != row["selected_uri"]
            or proposal.get("evidence_reference") != row["evidence_reference"]
            or row.get("reviewer") != m["approved_by"]
            or not row.get("rationale", "").startswith(
                f"Batch-approved AI proposal {m['proposal_sha256'][:12]} by {m['approved_by']}; "
            )
        ):
            raise ValueError(f"ConceptNet batch approval row mismatch at class {i}")


def _check_batch_review(m: dict, kg: KG, folder: Path) -> None:
    """Accept an exact, user-approved 150-synset selection without inventing rationales."""
    if kg.source != "wordnet" or m.get("review_basis") != "official_representative_batch_selection":
        raise ValueError("Batch approval only supports the official WordNet representative mapping")
    if (
        kg.metadata.get("resolved_class_count") != 150
        or kg.metadata.get("unresolved_class_ids") != []
        or kg.metadata.get("unreviewed_class_ids") != []
    ):
        raise ValueError("Batch approval requires 150 resolved and reviewed KG classes")
    files = {}
    for key in ("mapping", "overrides", "selections", "approval"):
        path = Path(m.get(f"{key}_path", ""))
        if not path.is_absolute() or not path.is_file() or sha256(path) != m.get(f"{key}_sha256"):
            raise ValueError(f"Batch review {key} path/hash mismatch")
        files[key] = path
    if kg.metadata.get("input_mapping_sha256") != m["mapping_sha256"]:
        raise ValueError("Batch review mapping/KG SHA256 mismatch")
    mapping = json.loads(files["mapping"].read_text(encoding="utf-8"))
    overrides = json.loads(files["overrides"].read_text(encoding="utf-8"))
    approval = json.loads(files["approval"].read_text(encoding="utf-8"))
    selection_lines = files["selections"].read_text(encoding="utf-8-sig").splitlines()
    if mapping.get("overrides_sha256") != m["overrides_sha256"]:
        raise ValueError("Batch review override/mapping SHA256 mismatch")
    if (
        approval.get("schema_version") != 1
        or approval.get("approval_scope") != "all_150_representative_synsets"
        or approval.get("reviewer") != m.get("reviewer")
        or not approval.get("review_date")
        or approval.get("review_date") != m.get("review_date")
        or not approval.get("statement")
        or approval.get("per_class_rationale_provided") is not False
        or m.get("per_class_rationale_provided") is not False
    ):
        raise ValueError("Batch review approval declaration is incomplete")
    reviewer = approval["reviewer"]
    if (
        not isinstance(reviewer, str)
        or not reviewer.strip()
        or "AI" in reviewer.upper()
        or "CHATGPT" in reviewer.upper()
    ):
        raise ValueError("Batch review requires a named human reviewer")
    rows = mapping.get("rows")
    reviews = m.get("reviews")
    if (
        not isinstance(rows, list)
        or len(rows) != 151
        or not isinstance(reviews, list)
        or len(reviews) != 150
        or len(selection_lines) != 151
        or selection_lines[0] != "selected_synset"
    ):
        raise ValueError("Batch review requires exactly 150 ordered selections")
    explicit = json.loads((folder / "class_entity_ids.json").read_text(encoding="utf-8")).get("class_entity_ids")
    if not isinstance(explicit, list) or len(explicit) != 151 or explicit[0] is not None:
        raise ValueError("Batch review class/entity alignment is invalid")
    labels, provenance = overrides.get("mappings"), overrides.get("review_provenance")
    if not isinstance(labels, dict) or not isinstance(provenance, dict):
        raise ValueError("Batch review override lacks mappings/provenance")
    for i in range(1, 151):
        row, item = rows[i], reviews[i - 1]
        synset = selection_lines[i].strip()
        label = row.get("label")
        audit = provenance.get(label, {})
        if (
            row.get("class_id") != i
            or row.get("status") != "linked_manual"
            or not synset
            or row.get("synset_id") != synset
            or labels.get(label) != synset
            or not isinstance(audit, dict)
            or audit.get("class_id") != i
            or audit.get("reviewer") != reviewer
            or not isinstance(item, dict)
            or item.get("class_id") != i
            or item.get("label") != label
            or item.get("selected_synset") != synset
            or item.get("reviewer") != reviewer
            or item.get("evidence_reference") != f"{files['selections']}#L{i + 1}"
        ):
            raise ValueError(f"Batch review selection/provenance mismatch at class {i}")
        entity_id = explicit[i]
        if (
            type(entity_id) is not int
            or not 0 <= entity_id < len(kg.entities)
            or kg.entities[entity_id] != f"wn:{synset}"
        ):
            raise ValueError(f"Batch review KG entity mismatch at class {i}")


class TransE(nn.Module):
    def __init__(self, n_entities: int, n_relations: int, dim: int = 64, margin: float = 1.0, p: int = 1):
        super().__init__()
        if dim <= 0 or margin <= 0 or p not in (1, 2):
            raise ValueError("Invalid TransE hyperparameters")
        self.entity = nn.Embedding(n_entities, dim)
        self.relation = nn.Embedding(n_relations, dim)
        self.margin = margin
        self.p = p
        bound = 6 / np.sqrt(dim)
        nn.init.uniform_(self.entity.weight, -bound, bound)
        nn.init.uniform_(self.relation.weight, -bound, bound)
        with torch.no_grad():
            self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=-1))

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        h, r, t = triples.unbind(dim=-1)
        return torch.linalg.vector_norm(self.entity(h) + self.relation(r) - self.entity(t), ord=self.p, dim=-1)

    def loss(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return F.relu(self.margin + self.score(positive) - self.score(negative)).mean()

    def normalize_entities(self) -> None:
        with torch.no_grad():
            self.entity.weight.copy_(F.normalize(self.entity.weight, p=2, dim=-1))


def negative_sample(
    positives: np.ndarray, n_entities: int, known: set[tuple[int, int, int]], rng: np.random.Generator
) -> np.ndarray:
    """Uniform head/tail corruption, filtered against caller-visible positives."""
    negatives = positives.copy()
    for i, (h, r, t) in enumerate(positives):
        # O(batch) expected rejection sampling; bounded fallback for dense graphs.
        chosen = None
        for _ in range(64):
            candidate_id = int(rng.integers(n_entities))
            corrupt_head = bool(rng.integers(2))
            candidate = (candidate_id, int(r), int(t)) if corrupt_head else (int(h), int(r), candidate_id)
            if candidate_id != (h if corrupt_head else t) and candidate not in known:
                chosen = candidate
                break
        if chosen is None:
            candidates = []
            for candidate_id in range(n_entities):
                if candidate_id != h and (candidate_id, int(r), int(t)) not in known:
                    candidates.append((candidate_id, int(r), int(t)))
                if candidate_id != t and (int(h), int(r), candidate_id) not in known:
                    candidates.append((int(h), int(r), candidate_id))
            if not candidates:
                raise ValueError("No valid negative candidate")
            chosen = candidates[int(rng.integers(len(candidates)))]
        negatives[i] = chosen
    return negatives


def split_triples(
    triples: np.ndarray, seed: int, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Invalid split ratios")
    if len(triples) < 10:
        raise ValueError("Need at least 10 unique triples for split")
    order = np.random.default_rng(seed).permutation(len(triples))
    nval = max(1, int(len(triples) * val_ratio))
    ntest = max(1, int(len(triples) * test_ratio))
    return triples[order[nval + ntest :]], triples[order[:nval]], triples[order[nval : nval + ntest]]


def split_triples_transductive(
    triples: np.ndarray, seed: int, val_ratio: float = 0.1, test_ratio: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coverage-preserving triple holdout: ALL evaluated entities/relations occur in train.

    Select essential edges in ascending endpoint-degree order, then reserve
    sufficient randomly chosen extra training edges to match the requested
    holdout sizes. Never silently change the requested ratios.
    """
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Invalid split ratios")
    triples = np.asarray(triples)
    if triples.ndim != 2 or triples.shape[1] != 3 or not np.issubdtype(triples.dtype, np.integer):
        raise ValueError("Triples must be an integer (N, 3) array")
    n = len(triples)
    if n < 10 or len({tuple(map(int, t)) for t in triples}) != n:
        raise ValueError("Need at least 10 distinct triples")
    nval, ntest = max(1, int(n * val_ratio)), max(1, int(n * test_ratio))
    train_budget = n - nval - ntest
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    # Insertion order is seed-randomized for reproducible tie breaking.
    incident: dict[tuple[str, int], list[int]] = {}
    degree: dict[tuple[str, int], int] = {}
    for i in order:
        h, r, t = map(int, triples[i])
        for key in {("e", h), ("e", t), ("r", r)}:
            incident.setdefault(key, []).append(int(i))
            degree[key] = degree.get(key, 0) + 1
    chosen: set[int] = set()
    covered: set[tuple[str, int]] = set()
    # Degree-1 nodes are mandatory; choosing constrained nodes first also
    # reduces the chance of over-allocating the training split.
    keys = sorted(incident, key=lambda x: degree[x])
    for key in keys:
        if key in covered:
            continue
        candidates = incident[key]

        def gain(i: int) -> int:
            h, r, t = map(int, triples[i])
            return sum(k not in covered for k in {("e", h), ("e", t), ("r", r)})

        index = max(candidates, key=gain)
        chosen.add(index)
        h, r, t = map(int, triples[index])
        covered.update({("e", h), ("e", t), ("r", r)})
        if len(chosen) > train_budget:
            raise ValueError(
                f"Transductive split infeasible at requested ratios: "
                f"coverage needs >{train_budget} training triples; lower holdout ratios"
            )
    remainder = [int(i) for i in order if int(i) not in chosen]
    extra_train = train_budget - len(chosen)
    train_ids = sorted(chosen) + remainder[:extra_train]
    holdout = remainder[extra_train:]
    train = triples[train_ids].copy()
    val = triples[holdout[:nval]].copy()
    test = triples[holdout[nval:]].copy()
    assert len(train) == train_budget and len(val) == nval and len(test) == ntest
    train_e = set(train[:, 0].tolist()) | set(train[:, 2].tolist())
    train_r = set(train[:, 1].tolist())
    for part in (val, test):
        if not (set(part[:, 0].tolist()) | set(part[:, 2].tolist())).issubset(train_e):
            raise AssertionError("Entity coverage invariant violated")
        if not set(part[:, 1].tolist()).issubset(train_r):
            raise AssertionError("Relation coverage invariant violated")
    return train, val, test


def evaluate(
    model: TransE, triples: np.ndarray, known: set[tuple[int, int, int]], device: torch.device, batch_size: int = 128
) -> dict:
    """Filtered raw-entity ranking on both head and tail, ties use average rank."""
    ranks = []
    # Build lookup once. The previous implementation copied every candidate
    # triple to CPU and launched ceil(entity_count/batch_size) tiny GPU batches
    # for *each* head and tail query. ConceptNet has ~46k entities, so score
    # all candidate entities in one tensor operation per query instead.
    head_known = {}
    tail_known = {}
    for h, r, t in known:
        head_known.setdefault((int(r), int(t)), set()).add(int(h))
        tail_known.setdefault((int(h), int(r)), set()).add(int(t))
    model.eval()
    with torch.no_grad():
        entities = model.entity.weight
        relations = model.relation.weight
        for h, r, t in triples:
            h, r, t = int(h), int(r), int(t)
            for side in (0, 2):
                diff = entities + relations[r] - entities[t] if side == 0 else entities[h] + relations[r] - entities
                scores = torch.linalg.vector_norm(diff, ord=model.p, dim=-1)
                gold_id = h if side == 0 else t
                gold = scores[gold_id]
                filtered = head_known.get((r, t), ()) if side == 0 else tail_known.get((h, r), ())
                filtered = [index for index in filtered if index != gold_id]
                if filtered:
                    scores[torch.as_tensor(filtered, dtype=torch.long, device=device)] = float("inf")
                better = int(torch.count_nonzero(scores < gold - 1e-8).item())
                tied = int(torch.count_nonzero(torch.abs(scores - gold) <= 1e-8).item())
                ranks.append(1 + better + (tied - 1) / 2)
    return {
        "MRR": float(np.mean([1 / x for x in ranks])),
        "Hits@1": float(np.mean([x <= 1 for x in ranks])),
        "Hits@3": float(np.mean([x <= 3 for x in ranks])),
        "Hits@10": float(np.mean([x <= 10 for x in ranks])),
        "count": len(ranks),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
