"""Official RelTR weights + frozen CLIP region attributes, no optimizer/training."""

import argparse
import ast
import importlib
import subprocess
import sys
from pathlib import Path

from src.editing.operations import EditGraph, Node

from .core import digest, sha256

RELTR_COMMIT = "fca7397e9aaeccd95541e83afa4b971f3fa89014"
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def verify_repo(repo):
    repo = Path(repo).resolve()
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != RELTR_COMMIT:
        raise ValueError("reltr_revision_mismatch")
    # Check file bytes against the pinned tree, including skip-worktree files.
    paths = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"], text=True
    ).splitlines()
    hashes = {}
    for name in paths:
        expected = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"])
        path = repo / name
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError("reltr_source_modified:" + name)
        hashes[name] = sha256(path)
    # Untracked modules could shadow imports from the reviewed source tree.
    for path in repo.rglob("*.py"):
        if path.relative_to(repo).as_posix() not in hashes:
            raise ValueError("untracked_reltr_python_source")
    return {"url": "https://github.com/yrcong/RelTR", "commit": commit, "files_sha256": hashes}


def vocabulary(repo):
    tree = ast.parse((Path(repo) / "inference.py").read_text())
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"CLASSES", "REL_CLASSES"}:
                result[name] = ast.literal_eval(node.value)
    if len(result.get("CLASSES", [])) != 151 or len(result.get("REL_CLASSES", [])) != 51:
        raise ValueError("unexpected_reltr_vocabulary")
    return result


def build_reltr(repo):
    """Use official modules; construct ResNet with weights=None (checkpoint is complete).

    Avoid upstream build(): it downloads redundant ImageNet weights and creates
    a training criterion. No external source is modified.
    """
    import torchvision

    repo = Path(repo).resolve()
    for name in ("models", "util"):
        loaded = sys.modules.get(name)
        if loaded and Path(loaded.__file__).resolve().parent != repo / name:
            raise RuntimeError("reltr_import_namespace_collision; use a fresh CLI process")
    sys.path.insert(0, str(repo))
    try:
        backbone = importlib.import_module("models.backbone")
        position = importlib.import_module("models.position_encoding")
        transformer = importlib.import_module("models.transformer")
        reltr = importlib.import_module("models.reltr")
    finally:
        sys.path.pop(0)
    cnn = torchvision.models.resnet50(weights=None, norm_layer=backbone.FrozenBatchNorm2d)
    body = backbone.BackboneBase(cnn, False, 2048, False)
    joined = backbone.Joiner(body, position.PositionEmbeddingSine(128, normalize=True))
    joined.num_channels = 2048
    trans = transformer.Transformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        return_intermediate_dec=True,
    )
    return reltr.RelTR(joined, trans, 151, 51, 100, 200, aux_loss=False).eval().requires_grad_(False)


def load_reltr(repo, checkpoint, expected_hash, device):
    import torch

    if sha256(checkpoint) != expected_hash:
        raise ValueError("reltr_checkpoint_hash_mismatch")
    model = build_reltr(repo)
    # The official checkpoint includes argparse.Namespace training metadata.
    # Never fall back to unrestricted pickle or execute that training config.
    with torch.serialization.safe_globals([argparse.Namespace]):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    return model.to(device)


def decode_outputs(outputs, size, vocab):
    """All entity queries plus relation endpoints; background/no-object suppressed."""
    import torch

    w, h = size
    detections, relations, dropped = [], [], []
    labels = vocab["CLASSES"]
    predicates = vocab["REL_CLASSES"]
    probabilities = {}
    for role, logits, boxes in (
        ("entity", "pred_logits", "pred_boxes"),
        ("sub", "sub_logits", "sub_boxes"),
        ("obj", "obj_logits", "obj_boxes"),
    ):
        scores = outputs[logits].detach().float().cpu()
        coords = outputs[boxes].detach().float().cpu()
        if (
            scores.ndim != 3
            or scores.shape[0] != 1
            or scores.shape[-1] != len(labels) + 1
            or coords.shape != (*scores.shape[:2], 4)
        ):
            raise ValueError("unexpected_reltr_output_shape")
        if not torch.isfinite(scores).all() or not torch.isfinite(coords).all():
            raise ValueError("nonfinite_reltr_output")
        probabilities[role] = scores.softmax(-1)[0]
        for i, (p, b) in enumerate(zip(probabilities[role], coords[0], strict=True)):
            index = int(p.argmax())
            identifier = f"{role}-{i:03d}"
            if index in (0, len(labels)):
                dropped.append({"id": identifier, "reason": "background_or_no_object"})
                continue
            cx, cy, bw, bh = b.tolist()
            box = [
                max(0.0, min(w, (cx - bw / 2) * w)),
                max(0.0, min(h, (cy - bh / 2) * h)),
                max(0.0, min(w, (cx + bw / 2) * w)),
                max(0.0, min(h, (cy + bh / 2) * h)),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                dropped.append({"id": identifier, "reason": "degenerate_box"})
                continue
            detections.append(
                {
                    "id": identifier,
                    "label": labels[index],
                    "class_index": index,
                    "bbox_xyxy": box,
                    "confidence": float(p[index]),
                }
            )
    valid = {d["id"] for d in detections}
    scores = outputs["rel_logits"].detach().float().cpu()
    if scores.shape != (1, len(probabilities["sub"]), len(predicates) + 1) or not torch.isfinite(scores).all():
        raise ValueError("unexpected_reltr_relation_output")
    for i, p in enumerate(scores.softmax(-1)[0]):
        index = int(p.argmax())
        if index in (0, len(predicates)) or f"sub-{i:03d}" not in valid or f"obj-{i:03d}" not in valid:
            dropped.append({"id": f"rel-{i:03d}", "reason": "background_or_missing_endpoint"})
            continue
        confidence = float(p[index])
        relations.append(
            {
                "id": f"rel-{i:03d}",
                "subject": f"sub-{i:03d}",
                "object": f"obj-{i:03d}",
                "predicate": predicates[index],
                "class_index": index,
                "confidence": confidence,
                "triplet_score": confidence
                * float(probabilities["sub"][i].max())
                * float(probabilities["obj"][i].max()),
            }
        )
    return {"detections": detections, "relations": relations, "dropped": dropped}


def infer_reltr(model, image, vocab, device, short_side):
    import torch
    from torchvision import transforms as T

    transform = T.Compose(
        [T.Resize(short_side), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    with torch.inference_mode():
        output = model(transform(image).unsqueeze(0).to(device))
    raw = decode_outputs(output, image.size, vocab)
    raw["model_outputs"] = {k: v.detach().float().cpu().tolist() for k, v in output.items()}
    return raw


def load_clip(weights, device):
    import clip

    if sha256(weights) != CLIP_SHA256:
        raise ValueError("clip_checkpoint_hash_mismatch")
    model, preprocess = clip.load(str(weights), device=device, jit=False)
    return model.eval().requires_grad_(False), preprocess


def select_attributes(cosines, labels, minimum, margin):
    """Cosine ranking with abstention, not a calibrated attribute classifier."""
    ranked = sorted(zip(labels, cosines, strict=True), key=lambda x: (-x[1], x[0]))
    if any(not -1 <= v <= 1 for _, v in ranked):
        raise ValueError("invalid_clip_cosine")
    accepted = ranked[0][1] >= minimum and ranked[0][1] - ranked[1][1] >= margin
    return ranked[0][0] if accepted else None


def add_attributes(model, preprocess, image, graph, grounding, config, device):
    import clip
    import torch
    import torch.nn.functional as F

    result, evidence = [], {}
    with torch.inference_mode():
        for node in graph.nodes:
            box = grounding["nodes"][node.id]["bbox_xyxy"]
            crop = image.crop(box)
            if crop.width < 1 or crop.height < 1:
                raise ValueError("empty_attribute_crop")
            visual = F.normalize(model.encode_image(preprocess(crop).unsqueeze(0).to(device)).float(), dim=-1)
            accepted, candidates = [], []
            for family, labels in config["attribute_vocabulary"].items():
                prompts = [f"a photo of a {a} {node.label}." for a in labels]
                text = F.normalize(model.encode_text(clip.tokenize(prompts).to(device)).float(), dim=-1)
                cosine = (visual @ text.T).squeeze(0).clamp(-1, 1).cpu().tolist()
                selected = select_attributes(
                    cosine, labels, config["attribute_min_cosine"], config["attribute_min_margin"]
                )
                for label, score, prompt in zip(labels, cosine, prompts, strict=True):
                    item = {
                        "label": label,
                        "family": family,
                        "cosine_similarity": score,
                        "confidence": (score + 1) / 2,
                        "confidence_kind": "rescaled_cosine_not_probability",
                        "prompt": prompt,
                        "origin": "clip_vit_b32_region",
                        "selected": label == selected,
                    }
                    candidates.append(item)
                    if label == selected:
                        accepted.append(item)
            grounding["nodes"][node.id]["attributes"] = accepted
            evidence[node.id] = candidates
            result.append(Node(node.id, node.label, tuple(a["label"] for a in accepted)))
    return EditGraph(tuple(result), graph.edges), evidence


def model_identity(repo, checkpoint, clip_weights, expected_hash):
    upstream = verify_repo(repo)
    if sha256(checkpoint) != expected_hash or sha256(clip_weights) != CLIP_SHA256:
        raise ValueError("pretrained_weight_hash_mismatch")
    vocab = vocabulary(repo)
    return {
        "reltr": {"source": upstream, "checkpoint_sha256": expected_hash, "vocabulary_sha256": digest(vocab)},
        "attributes": {"model": "OpenAI CLIP ViT-B/32", "checkpoint_sha256": CLIP_SHA256},
        "frozen": True,
        "training": False,
    }, vocab
