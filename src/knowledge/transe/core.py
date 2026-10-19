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
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    if m.get("schema_version") != 1 or m.get("review_status") != "human_approved" or m.get("kg_source") != kg.source:
        raise ValueError("Review manifest missing human approval or incorrect source")
    if m.get("kg_sha256") != kg.hashes or m.get("kg_path") != str(folder.resolve()):
        raise ValueError("Review manifest does not bind all exact input files/path")
    if kg.metadata.get("ready_for_gat_kg_training") is not True:
        raise ValueError("Source KG metadata explicitly not training-ready")
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
        if "AI" in item["reviewer"].upper() or "CHATGPT" in item["reviewer"].upper():
            raise ValueError("AI reviewer is not human review evidence")
    if (
        kg.metadata.get("curation_tier", "").upper().find("PROVISIONAL") >= 0
        or kg.metadata.get("human_verified_class_count") == 0
    ):
        raise ValueError("Provisional/unverified KG may not run in C-Reviewed")


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
    """Uniform head/tail corruption, filtered against ALL known true KG triples."""
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


def evaluate(
    model: TransE, triples: np.ndarray, known: set[tuple[int, int, int]], device: torch.device, batch_size: int = 128
) -> dict:
    """Filtered raw-entity ranking on both head and tail, ties use average rank."""
    ranks = []
    model.eval()
    with torch.no_grad():
        for h, r, t in triples:
            for side in (0, 2):
                query = np.tile([h, r, t], (model.entity.num_embeddings, 1))
                query[:, side] = np.arange(model.entity.num_embeddings)
                scores = []
                for chunk in np.array_split(query, max(1, (len(query) + batch_size - 1) // batch_size)):
                    scores.extend(model.score(torch.as_tensor(chunk, dtype=torch.long, device=device)).cpu().tolist())
                gold = scores[int(h if side == 0 else t)]
                for index in range(len(scores)):
                    triple = (int(index), int(r), int(t)) if side == 0 else (int(h), int(r), int(index))
                    if triple in known and triple != (int(h), int(r), int(t)):
                        scores[index] = float("inf")
                better = sum(s < gold - 1e-8 for s in scores)
                tied = sum(abs(s - gold) <= 1e-8 for s in scores)
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
