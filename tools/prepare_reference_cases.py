"""Prepare deterministic 5 hit / 5 miss validation cases for human review."""

import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.evaluate_level1 import load_model
from scripts.extract_level1_examples import predict_graph
from scripts.train_mlp import make_dataset
from src.evaluation.reference_predcls import PROTOCOL, box_iou, evaluate_reference_image
from src.knowledge.transe.core import sha256


def main():
    torch.set_num_threads(1)
    root = Path("data/raw/vg150")
    out = Path("reports/level1/submission/cases")
    if (out / "cases.json").exists():
        raise FileExistsError(out / "cases.json")
    out.mkdir(parents=True, exist_ok=True)
    cp = Path("runs/level1-clip-v1/clip-gat-seed42-v1/run_0001/best.pt")
    checkpoint = torch.load(cp, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = load_model(checkpoint).eval()
    dataset = make_dataset(root, "val", config["box_policy"], config["visual_cache"])
    vocab = json.loads((root / "VG-SGG-dicts-with-attri.json").read_text())
    objects = {v: k for k, v in vocab["label_to_idx"].items()}
    predicates = {v: k for k, v in vocab["predicate_to_idx"].items()}
    order = list(range(100))
    random.Random(2026).shuffle(order)
    cases = {"hit": [], "miss": []}
    for index in order:
        g = dataset[index]
        with torch.inference_mode():
            logits = predict_graph(model, config["model"], g, torch.device("cpu"))
        scores, labels = logits.softmax(-1)[:, 1:].max(-1)
        labels += 1
        ranked = scores.sort(descending=True).indices
        pairs = g.edge_label_index[:, ranked]
        gs, go = g.gt_relation_index
        ps, po = pairs
        iou = box_iou(g.boxes.float())
        obj = g.object_labels
        matches = (
            (obj[gs, None] == obj[ps][None, :])
            & (obj[go, None] == obj[po][None, :])
            & (g.gt_predicate[:, None] == labels[ranked][None, :])
            & (iou[gs[:, None], ps[None, :]] >= 0.5)
            & (iou[go[:, None], po[None, :]] >= 0.5)
        )
        hit = matches[:, :50].any(1)
        assert (
            int(hit.sum())
            == evaluate_reference_image(logits, g.edge_label_index, g.gt_relation_index, g.gt_predicate, g.boxes, obj)[
                "hits"
            ][50]
        )
        metadata = dataset.loader.image_metadata[dataset.loader.indices[index]]
        for kind in cases:
            if len(cases[kind]) >= 5:
                continue
            ids = torch.where(hit if kind == "hit" else ~hit)[0]
            if not len(ids):
                continue
            row = int(ids[0])
            s, o = int(gs[row]), int(go[row])
            predicate = int(g.gt_predicate[row])
            same_pair_rank = int(torch.where((pairs[0] == s) & (pairs[1] == o))[0]) + 1
            matches_at = torch.where(matches[row])[0]
            first_match = int(matches_at[0]) + 1 if len(matches_at) else None
            predicted_id = int(labels[ranked[same_pair_rank - 1]])
            image_path = root / "images" / f"{g.image_id}.jpg"
            key = f"{kind}{len(cases[kind])+1:02d}"
            record = {
                "case_id": key,
                "kind": kind,
                "validation_index": index,
                "image_id": str(g.image_id),
                "gt_annotation_row": row,
                "subject_node": s,
                "object_node": o,
                "subject_label": objects[int(obj[s])],
                "object_label": objects[int(obj[o])],
                "gt_predicate": predicates[predicate],
                "same_pair_prediction": predicates[predicted_id],
                "same_pair_rank": same_pair_rank,
                "first_reference_match_rank": first_match,
                "reason": "matched_in_top50"
                if kind == "hit"
                else "matching_prediction_below_top50"
                if first_match
                else "no_matching_prediction",
                "subject_box": g.boxes[s].tolist(),
                "object_box": g.boxes[o].tolist(),
                "annotation_size": [metadata["width"], metadata["height"]],
                "image_sha256": sha256(image_path),
                "image_url": metadata.get("url"),
                "image_path": str(image_path),
                "panel": f"{key}.png",
                "human_review_status": "pending",
                "human_reviewer": "",
                "human_analysis": "",
            }
            image = Image.open(image_path).convert("RGB")
            record["jpeg_size"] = list(image.size)
            draw = ImageDraw.Draw(image)
            for node, color in [(s, "red"), (o, "blue")]:
                box = g.boxes[node].tolist()
                sx = image.width / metadata["width"]
                sy = image.height / metadata["height"]
                box = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                draw.rectangle(box, outline=color, width=4)
                draw.text(
                    (box[0] + 2, box[1] + 2),
                    f"{node}: {objects[int(obj[node])]}",
                    fill="white",
                    stroke_width=2,
                    stroke_fill="black",
                )
            image.thumbnail((720, 440))
            panel = Image.new("RGB", (760, 540), "white")
            panel.paste(image, ((760 - image.width) // 2, 80))
            d = ImageDraw.Draw(panel)
            font = ImageFont.load_default(size=17)
            d.text((15, 8), f"{key} | image {g.image_id} | red=subject, blue=object", fill="black", font=font)
            d.text(
                (15, 31),
                f"GT: {record['subject_label']} --{record['gt_predicate']}--> {record['object_label']}",
                fill="black",
                font=font,
            )
            d.text(
                (15, 54),
                f"Same-pair prediction: {record['same_pair_prediction']} | rank {same_pair_rank}",
                fill="black",
                font=font,
            )
            panel.save(out / record["panel"])
            cases[kind].append(record)
        if all(len(v) == 5 for v in cases.values()):
            break
    assert all(len(v) == 5 for v in cases.values())
    result = {
        "protocol": PROTOCOL,
        "checkpoint": str(cp),
        "checkpoint_sha256": sha256(cp),
        "model": "gat",
        "seed": 42,
        "split": "val",
        "selection_seed": 2026,
        "pool_size": 100,
        "selection": "shuffle first 100 validation images; first GT annotation per category per image; stop at five each; no visual cherry-picking",
        "scope": "illustrative validation cases; not a prevalence estimate; human review pending",
        "cases": cases,
    }
    (out / "cases.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for kind in cases:
        sheet = Image.new("RGB", (1520, 1620), "#dddddd")
        for i, c in enumerate(cases[kind]):
            sheet.paste(Image.open(out / c["panel"]), ((i % 2) * 760, (i // 2) * 540))
        sheet.save(out / f"{kind}-contact-sheet.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
