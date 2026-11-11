"""Create AI-provisional ConceptNet class-link suggestions for human approval.

This is a navigation aid. It does not modify review.csv, assign a human
reviewer, or make the ConceptNet graph training-ready.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from nltk.corpus import wordnet

from scripts.build_conceptnet_graph import write_outputs
from scripts.prepare_conceptnet_review import check

EXPECTED_MAPPING_SHA256 = "c6b38d444b323f197d001bb92f8b62e81e14b920821f227b046ecdcd8b54fef8"

# These cases need attention beyond an exact lexical match. Reasons are about
# the local ConceptNet 5.7 evidence and the representative VG150 WordNet sense.
SPECIAL = {
    11: ("defer", "VG representative is motorcycle, but bare bike evidence also describes bicycles."),
    21: ("defer", "Branch mixes a tree limb with organization and control-structure senses."),
    25: ("defer", "Cap can mean headwear or a cover; evidence does not settle the VG object sense."),
    32: ("defer", "Counter mixes a kitchen/shop surface with a counting device or action."),
    54: ("defer", "Bare glass mixes drinking vessel and material; /n evidence describes material."),
    59: ("defer", "Handle mixes a physical grip with the verb handle."),
    68: ("defer", "Kid mixes human child and young goat in the retained assertions."),
    67: ("defer", "Bare jean has little useful garment evidence; /n may describe denim fabric."),
    75: ("defer", "Letter mixes an alphabet symbol with written correspondence."),
    76: ("defer", "Light mixes a lighting device with light as energy/property."),
    86: ("defer", "Orange mixes fruit and color in the retained assertions."),
    85: ("defer", "Number mixes a written numeral, quantity and phone number."),
    87: ("reject", "Available pant URI is used chiefly as a verb, not trousers."),
    89: ("defer", "Paw has only weak retained lexical evidence for the intended animal foot."),
    95: ("defer", "Plane can mean aircraft or a flat geometric surface."),
    96: ("defer", "Plant can mean organism or industrial facility; compare the candidate graphs."),
    100: ("defer", "Post can mean an upright support, mail, or a job."),
    101: ("defer", "VG representative is a plant pot; bare pot evidence mainly describes cooking vessels."),
    104: ("defer", "Rock mixes mineral matter and music."),
    107: ("defer", "VG representative is an electronic display; screen evidence also concerns cinema/partitions."),
    98: ("defer", "Player mixes sport participants and video-game players."),
    113: ("reject", "VG representative is short pants; available URIs describe adjective short or a film."),
    118: ("defer", "Ski evidence includes unrelated brand/word senses; inspect the noun URI."),
    123: ("defer", "VG representative is a support/base; stand evidence mixes verb, booth and furniture."),
    125: ("defer", "Surfboard has sparse direct evidence for the bare URI."),
    127: ("defer", "Tail evidence mixes animal body part and other tail-shaped objects."),
    138: ("defer", "VG representative is a tree trunk; evidence mixes car and elephant trunks."),
    143: ("defer", "Wave may be a water ridge or gesture/other oscillation."),
    147: ("defer", "Wing may be a bird wing, aircraft part, or building wing."),
}

# Resolved from inspection of every retained edge touching each candidate URI.
# Keep these separate from the generic lexical suggestions so the reason and
# supporting assertion for each changed decision stay reviewable.
RESOLVED = {
    21: (
        "branch",
        "/c/en/branch",
        ("RelatedTo", "/c/en/tree", "out"),
        "Tree, limb and twig are the strongest neighbours; /n is an organization branch.",
    ),
    25: (
        "cap",
        "/c/en/cap/n",
        ("IsA", "/c/en/headgear/n", "out"),
        "The noun URI is typed as headgear with several hat subtypes.",
    ),
    59: (
        "handle",
        "/c/en/handle",
        ("PartOf", "/c/en/door", "out"),
        "Physical grips dominate the structural and use edges (door, mug, tool).",
    ),
    86: (
        "orange",
        "/c/en/orange/n",
        ("IsA", "/c/en/citrus_fruit/n", "out"),
        "The noun URI has a citrus-fruit type; the bare URI strongly mixes fruit and colour.",
    ),
    95: (
        "plane",
        "/c/en/plane",
        ("CapableOf", "/c/en/arrive_at_airport", "out"),
        "Aircraft and travel edges dominate; /n is explicitly a geometric surface.",
    ),
    96: (
        "plant",
        "/c/en/plant",
        ("IsA", "/c/en/organism", "out"),
        "Organism, garden and botanical edges dominate the bare URI.",
    ),
    98: (
        "player",
        "/c/en/player",
        ("PartOf", "/c/en/team", "out"),
        "A game or sport participant matches the VG sense; /n also mixes actor and instrument senses.",
    ),
    100: (
        "post",
        "/c/en/post/n",
        ("IsA", "/c/en/supporting_structure/n", "out"),
        "The noun URI is typed as a physical support; the bare URI favours mail.",
    ),
    107: (
        "screen",
        "/c/en/screen",
        ("PartOf", "/c/en/monitor", "out"),
        "The bare URI connects directly to monitor and TV display surfaces.",
    ),
    118: (
        "ski",
        "/c/en/ski",
        ("RelatedTo", "/c/en/snow", "out"),
        "Most neighbours concern skis and snow sports; one low-weight yoghurt-brand edge remains.",
    ),
    125: (
        "surfboard",
        "/c/en/surfboard/n",
        ("IsA", "/c/en/riding_board/n", "out"),
        "The noun URI is typed as a riding board and a water transportation device.",
    ),
    127: (
        "tail",
        "/c/en/tail",
        ("PartOf", "/c/en/cat", "out"),
        "Animal anatomy dominates the structural edges; a few kite-tail edges remain.",
    ),
    147: (
        "wing",
        "/c/en/wing",
        ("PartOf", "/c/en/bird", "out"),
        "Bird and aircraft flight appendages dominate; the building-wing sense is weaker.",
    ),
}

# Pick an interpretable, source-backed assertion for the draft where a generic
# score otherwise selects an unrelated high-weight assertion.
EVIDENCE_HINTS = {
    3: ("RelatedTo", "/c/en/hand", "out"),
    30: ("RelatedTo", "/c/en/time", "out"),
    33: ("IsA", "/c/en/hoofed_mammal", "out"),
    41: ("HasA", "/c/en/trunk", "out"),
    44: ("HasA", "/c/en/nose", "out"),
    57: ("AtLocation", "/c/en/head", "out"),
    71: ("UsedFor", "/c/en/illuminating_area", "out"),
    92: ("UsedFor", "/c/en/communication", "out"),
    108: ("IsA", "/c/en/chair", "out"),
    119: ("AtLocation", "/c/en/mountain", "out"),
    130: ("MadeOf", "/c/en/rubber", "out"),
    135: ("HasA", "/c/en/carriages", "out"),
    139: ("UsedFor", "/c/en/protect_against_rain", "out"),
    141: ("IsA", "/c/en/food", "out"),
    142: ("UsedFor", "/c/en/mobility", "out"),
    146: ("PartOf", "/c/en/car", "out"),
}
STOP = {"a", "an", "and", "as", "at", "be", "by", "for", "in", "is", "of", "on", "or", "the", "to", "with"}
RELATION_SCORE = {
    "IsA": 5,
    "UsedFor": 4,
    "HasA": 4,
    "PartOf": 3,
    "AtLocation": 2,
    "MadeOf": 2,
    "RelatedTo": 1,
    "HasProperty": 1,
    "CapableOf": 1,
    "ReceivesAction": 1,
}
FIELDS = (
    "class_id",
    "label",
    "priority",
    "representative_wordnet_synset",
    "wordnet_gloss",
    "conceptnet_uri_candidates",
    "proposed_decision",
    "proposed_uri",
    "evidence_reference",
    "evidence_summary",
    "ai_rationale",
    "proposed_by",
    "requested_human_approver",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokens(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z]+", text.casefold().replace("_", " ")) if len(word) > 2 and word not in STOP
    }


def _all_evidence(kg_dir: Path, candidates: dict[int, list[str]]) -> dict[str, list[dict]]:
    entities = json.loads((kg_dir / "entities.json").read_text(encoding="utf-8"))["entities"]
    relations = json.loads((kg_dir / "relations.json").read_text(encoding="utf-8"))["relations"]
    wanted = {f"cn:{uri}" for uris in candidates.values() for uri in uris}
    result = defaultdict(list)
    with (kg_dir / "triples.tsv").open(encoding="utf-8", newline="") as stream:
        for edge in csv.DictReader(stream, delimiter="\t"):
            subject = entities[int(edge["subject_id"])]
            object_ = entities[int(edge["object_id"])]
            for concept, direction, other in ((subject, "out", object_), (object_, "in", subject)):
                if concept in wanted:
                    result[concept.removeprefix("cn:")].append(
                        {
                            "direction": direction,
                            "relation": relations[int(edge["relation_id"])],
                            "other_uri": other.removeprefix("cn:"),
                            "assertion_uri": edge["assertion_uri"],
                            "dataset": edge["dataset"],
                            "weight": float(edge["weight"]),
                        }
                    )
    return result


def _best_edge(edges: list[dict], gloss: str, synset) -> dict | None:
    if not edges:
        return None
    context = _tokens(gloss)
    for parent in synset.hypernyms():
        context |= _tokens(" ".join(parent.lemma_names()))

    def score(edge: dict):
        other = edge["other_uri"].split("/c/en/")[-1].split("/")[0]
        overlap = len(_tokens(other) & context)
        # Source weight is a small tie breaker, not a truth probability.
        return (
            10 * overlap
            + RELATION_SCORE.get(edge["relation"], 0)
            + (2 if edge["direction"] == "out" else 0)
            + min(math.log1p(edge["weight"]), 2),
            -len(other),
            edge["assertion_uri"],
        )

    return max(edges, key=score)


def build(kg_dir: Path, review_dir: Path, mapping_path: Path, *, approver: str) -> tuple[dict[str, bytes], dict]:
    if check(kg_dir, review_dir) != {"accept": 0, "reject": 0, "defer": 0, "pending": 150}:
        raise ValueError("AI suggestions require an untouched, pending human review worksheet")
    if _sha256(mapping_path) != EXPECTED_MAPPING_SHA256:
        raise ValueError("AI suggestion reasons are pinned to Toan's exact WordNet representative mapping")
    with (review_dir / "review.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["rows"]
    if len(rows) != 150 or len(mapping) != 151:
        raise ValueError("Expected 150 ConceptNet rows and 151 WordNet representative rows")
    if any(row["class_id"] != str(i) or row["label"] != mapping[i]["label"] for i, row in enumerate(rows, 1)):
        raise ValueError("ConceptNet and WordNet representative labels/IDs differ")
    candidates = {int(row["class_id"]): row["conceptnet_uri_candidates"].split(";") for row in rows}
    evidence = _all_evidence(kg_dir, candidates)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    focus_stream = io.StringIO(newline="")
    focus_writer = csv.DictWriter(focus_stream, fieldnames=FIELDS, lineterminator="\n")
    focus_writer.writeheader()
    counts = Counter()
    for row in rows:
        class_id = int(row["class_id"])
        synset_id = mapping[class_id]["synset_id"]
        synset = wordnet.synset(synset_id)
        gloss = synset.definition()
        uris = candidates[class_id]
        decision, special_reason = SPECIAL.get(class_id, ("accept", ""))
        resolved = RESOLVED.get(class_id)
        if resolved and row["label"] == resolved[0]:
            if resolved[1] not in uris:
                raise ValueError(f"Resolved URI is not a candidate at class {class_id}")
            decision, special_reason = "accept", resolved[3]
            proposed_uri = resolved[1]
        else:
            resolved = None
            proposed_uri = uris[0] if decision == "accept" else ""
        edge = _best_edge(evidence.get(proposed_uri or uris[0], []), gloss, synset)
        hint = resolved[2] if resolved else EVIDENCE_HINTS.get(class_id)
        if hint and proposed_uri:
            matching = [
                item
                for item in evidence[proposed_uri]
                if (item["relation"], item["other_uri"], item["direction"]) == hint
            ]
            if not matching:
                raise ValueError(f"Missing pinned ConceptNet evidence hint for class {class_id}")
            edge = max(matching, key=lambda item: (item["weight"], item["assertion_uri"]))
        if decision == "accept" and edge is None:
            decision, special_reason, proposed_uri = "defer", "No retained assertion supports the selected URI.", ""
        counts[decision] += 1
        summary = f"{edge['direction']} {edge['relation']} {edge['other_uri']} [{edge['dataset']}]" if edge else "none"
        rationale = (
            f"AI provisional: {special_reason} Intended VG sense: {synset_id} ({gloss}). Inspect both candidate graphs."
            if decision != "accept"
            else f"AI provisional after full local candidate-edge inspection: {special_reason} "
            f"Selected {proposed_uri} for VG sense {synset_id} ({gloss}); supporting {summary}."
            if resolved
            else f"AI provisional: the bare ConceptNet term and sampled {summary} appear compatible with "
            f"the representative VG sense {synset_id}; check for competing senses before approval."
        )
        proposal = {
            "class_id": class_id,
            "label": row["label"],
            "priority": "human_focus" if class_id in SPECIAL else row["priority"],
            "representative_wordnet_synset": synset_id,
            "wordnet_gloss": gloss,
            "conceptnet_uri_candidates": row["conceptnet_uri_candidates"],
            "proposed_decision": decision,
            "proposed_uri": proposed_uri,
            "evidence_reference": edge["assertion_uri"] if edge else "no_candidate_in_snapshot",
            "evidence_summary": summary,
            "ai_rationale": rationale,
            "proposed_by": "Codex_AI_provisional",
            "requested_human_approver": approver,
        }
        writer.writerow(proposal)
        if decision != "accept":
            focus_writer.writerow(proposal)
    provenance = {
        "schema_version": 1,
        "status": "AI_PROVISIONAL_NOT_HUMAN_REVIEWED_NOT_TRAINING_READY",
        "requested_human_approver": approver,
        "counts": dict(sorted(counts.items())),
        "human_focus_count": counts["defer"] + counts["reject"],
        "source_sha256": {
            "review.csv": _sha256(review_dir / "review.csv"),
            "wordnet_mapping.json": _sha256(mapping_path),
            "conceptnet_triples.tsv": _sha256(kg_dir / "triples.tsv"),
        },
    }
    return {
        "review_suggestions.ai-draft.csv": stream.getvalue().encode("utf-8"),
        "review_focus.ai-draft.csv": focus_stream.getvalue().encode("utf-8"),
        "review_suggestions.ai-draft.json": (json.dumps(provenance, indent=2) + "\n").encode("utf-8"),
    }, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-dir", type=Path, default=Path("data/knowledge/conceptnet-b2-v1"))
    parser.add_argument("--review-dir", type=Path, default=Path("data/knowledge/conceptnet-review-ai-v2"))
    parser.add_argument(
        "--mapping", type=Path, default=Path("data/knowledge/wordnet-reviewed-toan-v1/wordnet_mapping.json")
    )
    parser.add_argument("--approver", default="Toan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs, provenance = build(args.kg_dir, args.review_dir, args.mapping, approver=args.approver)
    write_outputs(args.review_dir, outputs, dry_run=args.dry_run)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
