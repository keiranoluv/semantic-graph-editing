"""GPU protocol v3, derived from frozen v2 orchestration to preserve CPU evidence.

Runtime differences: explicit device transfer, device-correct encoder mask,
CUDA deterministic settings and independent configuration/output/provenance.
Dataset/compiler, supervision, action vocabulary and evaluation remain v2.
"""

import argparse
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from common.editing.operations import EditGraph, apply_operations
from task2_graph_editing.magicbrush.sequence_model.codec import (
    EOS,
    canonical_actions,
    canonical_source,
    decode_actions,
    encode_actions,
    score_predictions,
)
from task2_graph_editing.magicbrush.sequence_model.model import collate_inputs
from common.io import digest, read, sha256, write
from task2_graph_editing.he2020.preprocessing.dataset import ARCHIVE, ROOT
from task2_graph_editing.he2020.models.model import LexicalPredictor
from task2_graph_editing.he2020.data_processing.data import load_bundle, prepare
from task2_graph_editing.he2020.data_processing.model import ActionVocabulary, frozen_features


def identity(config):
    paths = list((ROOT / "task2_graph_editing/he2020/data_processing").glob("*.py"))
    paths += list((ROOT / "task2_graph_editing/he2020/preprocessing").glob("*.py"))
    paths += list((ROOT / "task2_graph_editing/magicbrush/sequence_model").glob("*.py"))
    paths += list((ROOT / "common/editing").glob("*.py"))
    paths += list((ROOT / "task2_graph_editing/he2020/models").glob("*.py"))
    return {"config": config, "files": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}}


def tensor_subset(tensors, indices, device):
    return {key: value[indices].to(device) for key, value in tensors.items()}


def predict(model, tensors, vocabulary, config):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors["instruction"]), config["batch_size"]):
            batch = {k: v[start : start + config["batch_size"]].to(config["device"]) for k, v in tensors.items()}
            predictions.extend(vocabulary.to_bytes(row) for row in model.predict(batch, config["max_decode_tokens"]))
    return predictions


def evaluate(examples, predictions, vocabulary):
    metrics, errors = score_predictions(examples, predictions)
    oov = [3 in vocabulary.encode(e.operations) for e in examples]
    metrics["reference_programs_with_train_vocabulary_OOV"] = sum(oov)
    metrics["closed_vocabulary_exact_match_upper_bound"] = 1 - sum(oov) / len(oov) if oov else None
    metrics["by_reference_edit_kind"] = {}
    for kind in ("add_node", "delete_node", "replace_node", "replace_edge"):
        indices = [i for i, e in enumerate(examples) if any(o.kind == kind for o in e.operations)]
        if indices:
            metrics["by_reference_edit_kind"][kind] = score_predictions(
                [examples[i] for i in indices], [predictions[i] for i in indices]
            )[0]
    cases = []
    for e, tokens, error, unknown in zip(examples, predictions, errors, oov, strict=True):
        ops = ()
        try:
            ops = decode_actions(tokens, e.source)
            ops = canonical_actions(e.source, ops)
        except (ValueError, TypeError, KeyError, UnicodeError):
            pass
        exact = error is None and set(ops) == set(e.operations)
        cases.append(
            {
                "id": e.id,
                "correct": exact,
                "error": error,
                "reference_OOV": unknown,
                "instruction": e.instruction,
                "source": e.source.to_dict(),
                "target": e.target.to_dict(),
                "expected": [o.to_dict() for o in e.operations],
                "predicted": [o.to_dict() for o in ops],
            }
        )
    metrics["error_categories"] = dict(
        Counter("correct" if c["correct"] else (c["error"] or "valid_but_wrong").split(":")[0] for c in cases)
    )
    selected = []
    for correct in (True, False):
        selected += sorted((c for c in cases if c["correct"] == correct), key=lambda c: digest([42, c["id"]]))[:5]
    return metrics, cases, selected


def make_model(config, kind, vocabulary):
    return LexicalPredictor(
        kind=kind,
        vocabulary_size=vocabulary.size,
        text_dim=512,
        hidden=config["hidden"],
        byte_dim=48,
        layers=config["layers"],
        max_nodes=config["max_nodes"],
        max_edges=config["max_edges"],
    ).to(config["device"])


def inputs_tensor(examples, store, config):
    if any(len(e.source.nodes) > config["max_nodes"] or len(e.source.edges) > config["max_edges"] for e in examples):
        raise ValueError("source_identity_budget_exceeded")
    return collate_inputs(
        [e.model_input() for e in examples],
        store,
        max_nodes=max(1, max(len(e.source.nodes) for e in examples)),
        max_edges=max(1, max(len(e.source.edges) for e in examples)),
        device="cpu",
    )


def train_one(config, kind, seed, examples, tensors, vocabulary, experiment_identity):
    directory = Path(config["output"]) / f"{kind}-seed{seed}"
    if (directory / "result.json").exists():
        result = read(directory / "result.json")
        if (
            result["experiment_identity"] != experiment_identity
            or sha256(directory / "best.pt") != result["checkpoint_sha256"]
        ):
            raise ValueError("completed_run_identity_changed")
        return result
    random.seed(seed)
    torch.manual_seed(seed)
    if config["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = make_model(config, kind, vocabulary)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_function = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    targets = {
        split: tuple(t.to(config["device"]) for t in vocabulary.collate(rows)) for split, rows in examples.items()
    }
    best_loss, history = float("inf"), []
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(examples["train"]))
        total_loss = total_tokens = 0
        for indices in order.split(config["batch_size"]):
            previous, truth = (t[indices] for t in targets["train"])
            logits = model(tensor_subset(tensors["train"], indices, config["device"]), previous)
            count = int((truth != 0).sum())
            loss_sum = loss_function(logits.flatten(0, 1), truth.flatten())
            optimizer.zero_grad(set_to_none=True)
            (loss_sum / count).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss_sum.detach())
            total_tokens += count
        model.eval()
        val_loss = val_tokens = 0
        with torch.inference_mode():
            for start in range(0, len(examples["val"]), config["batch_size"]):
                indices = slice(start, start + config["batch_size"])
                previous, truth = (t[indices] for t in targets["val"])
                logits = model(tensor_subset(tensors["val"], indices, config["device"]), previous)
                val_loss += float(loss_function(logits.flatten(0, 1), truth.flatten()))
                val_tokens += int((truth != 0).sum())
        val_loss /= val_tokens
        row = {"epoch": epoch, "train_loss": total_loss / total_tokens, "val_loss": val_loss}
        history.append(row)
        print(
            json.dumps({"model": kind, "seed": seed, **row, "elapsed_seconds": round(time.monotonic() - started, 1)}),
            flush=True,
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "vocabulary": vocabulary.tokens,
                    "config": config,
                    "experiment_identity": experiment_identity,
                },
                directory / "best.pt",
            )
    checkpoint = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict(model, tensors["val"], vocabulary, config)
    metrics, cases, selected = evaluate(examples["val"], predictions, vocabulary)
    write(directory / "val-predictions.json", cases)
    write(directory / "val-cases.json", selected)
    # Paired diagnostic ablations use the same checkpoint and denominator.
    shuffled = {k: v.clone() if k == "instruction" else v for k, v in tensors["val"].items()}
    permutation = torch.randperm(len(examples["val"]), generator=torch.Generator().manual_seed(2020))
    shuffled["instruction"] = shuffled["instruction"][permutation]
    ablations = {
        "shuffled_instruction": score_predictions(examples["val"], predict(model, shuffled, vocabulary, config))[0]
    }
    if kind == "gnn":
        no_edges = {**tensors["val"], "edge_mask": torch.zeros_like(tensors["val"]["edge_mask"])}
        ablations["removed_edges"] = score_predictions(examples["val"], predict(model, no_edges, vocabulary, config))[0]
    result = {
        "model": kind,
        "seed": seed,
        "device": config["device"],
        "best_epoch": best_epoch,
        "history": history,
        "validation": metrics,
        "validation_ablations": ablations,
        "checkpoint_sha256": sha256(directory / "best.pt"),
        "experiment_identity": experiment_identity,
        "runtime_seconds": time.monotonic() - started,
        "test_used": False,
    }
    write(directory / "result.json", result)
    return result


def train(config):
    examples = load_bundle(config["dataset"], "trainval")
    if not examples["train"] or not examples["val"]:
        raise ValueError("empty_training_or_validation_split")
    if {e.group for e in examples["train"]} & {e.group for e in examples["val"]}:
        raise ValueError("source_graph_leakage")
    vocabulary = ActionVocabulary.fit(examples["train"])
    store = frozen_features(
        [e.model_input() for rows in examples.values() for e in rows],
        Path(config["dataset"]) / "clip-trainval",
        config["weights"],
        config["text_batch_size"],
    )
    tensors = {split: inputs_tensor(rows, store, config) for split, rows in examples.items()}
    experiment_identity = digest(
        [
            identity(config),
            sha256(Path(config["dataset"]) / "trainval-receipt.json"),
            sha256(Path(config["dataset"]) / "clip-trainval/manifest.json"),
            vocabulary.tokens,
        ]
    )
    write(
        Path(config["output"]) / "experiment.json",
        {
            "identity": experiment_identity,
            "source": identity(config),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "torch": torch.__version__,
            "device": config["device"],
            "gpu": torch.cuda.get_device_name() if config["device"] == "cuda" else None,
            "CUDA_runtime": torch.version.cuda,
            "TF32": False,
            "derived_from_frozen_CPU_protocol": "dd5c8b1",
            "continuation_authorization": "user_explicitly_requested_GPU_training_and_evaluation",
            "python": platform.python_version(),
            "trainval_receipt_sha256": sha256(Path(config["dataset"]) / "trainval-receipt.json"),
            "vocabulary": vocabulary.tokens,
            "vocabulary_origin": "train_only_plus_six_kinds_and_schema_ids",
            "test_used": False,
        },
    )
    runs = []
    for kind in config["models"]:
        for seed in config["seeds"]:
            runs.append(train_one(config, kind, seed, examples, tensors, vocabulary, experiment_identity))
    freeze = {
        "experiment_identity": experiment_identity,
        "runs": [
            {
                "model": r["model"],
                "seed": r["seed"],
                "checkpoint_sha256": r["checkpoint_sha256"],
                "result_sha256": sha256(Path(config["output"]) / f'{r["model"]}-seed{r["seed"]}' / "result.json"),
            }
            for r in runs
        ],
        "test_used": False,
    }
    write(Path(config["output"]) / "frozen-checkpoints.json", freeze)
    return freeze


def retrieval_baseline(config, examples, test_store):
    # Frozen features of training inputs only. Nearest training edit program is
    # decoded with BEFORE payloads from each new source, never from test targets.
    train_examples = load_bundle(config["dataset"], "trainval")["train"]
    train_store = frozen_features(
        [e.model_input() for rows in load_bundle(config["dataset"], "trainval").values() for e in rows],
        Path(config["dataset"]) / "clip-trainval",
        config["weights"],
        config["text_batch_size"],
    )
    from task2_graph_editing.magicbrush.sequence_model.data import feature_texts

    def vector(e, store):
        features = feature_texts(**e.model_input())
        graph_texts = features["nodes"] + features["edges"]
        graph = torch.stack([store[t] for t in graph_texts]).mean(0) if graph_texts else torch.zeros(512)
        return nn.functional.normalize(torch.cat((store[e.instruction], nn.functional.normalize(graph, dim=0))), dim=0)

    train_vectors = torch.stack([vector(e, train_store) for e in train_examples])
    test_vectors = torch.stack([vector(e, test_store) for e in examples])
    predictions = []
    for vectors in test_vectors.split(config["batch_size"]):
        nearest = (vectors @ train_vectors.T).argmax(-1).tolist()
        predictions.extend(encode_actions(train_examples[i].operations)[1:] for i in nearest)
    return score_predictions(examples, predictions)[0]


def infer(config, kind, seed, input_path, output_path):
    directory = Path(config["output"]) / f"{kind}-seed{seed}"
    result = read(directory / "result.json")
    if sha256(directory / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError("inference_checkpoint_changed")
    checkpoint = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    item = read(input_path)
    source, node_ids, edge_ids = canonical_source(EditGraph.from_dict(item["source_graph"]))
    model_input = {"source": source, "instruction": item["instruction"]}
    feature_id = digest([source.to_dict(), item["instruction"]])
    store = frozen_features(
        [model_input],
        Path(config["dataset"]) / "inference-features" / feature_id,
        config["weights"],
        config["text_batch_size"],
    )
    batch = collate_inputs([model_input], store, max_nodes=config["max_nodes"], max_edges=config["max_edges"])
    vocabulary = ActionVocabulary(checkpoint["vocabulary"])
    model = make_model(config, kind, vocabulary)
    model.load_state_dict(checkpoint["state_dict"])
    tokens = predict(model, batch, vocabulary, config)[0]
    output = {
        "checkpoint_sha256": result["checkpoint_sha256"],
        "source_graph": source.to_dict(),
        "input_node_ids": node_ids,
        "input_edge_ids": edge_ids,
        "instruction": item["instruction"],
        "operations": None,
        "target_graph": None,
        "error": None,
    }
    try:
        ops = canonical_actions(source, decode_actions(tokens, source))
        output.update(operations=[o.to_dict() for o in ops], target_graph=apply_operations(source, ops).to_dict())
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        output["error"] = str(exc)
    write(output_path, output)
    return {"output": str(output_path), "valid_transaction": output["error"] is None}


def final_test(config, archive):
    directory = Path(config["output"])
    freeze = read(directory / "frozen-checkpoints.json")
    experiment = read(directory / "experiment.json")
    if (
        identity(config) != experiment["source"]
        or freeze["experiment_identity"] != experiment["identity"]
        or sha256(Path(config["dataset"]) / "trainval-receipt.json") != experiment["trainval_receipt_sha256"]
    ):
        raise ValueError("code_or_config_changed_since_training")
    expected = {(k, s) for k in config["models"] for s in config["seeds"]}
    if {(r["model"], r["seed"]) for r in freeze["runs"]} != expected:
        raise ValueError("all_configured_checkpoints_must_be_frozen_before_test")
    for run in freeze["runs"]:
        path = directory / f'{run["model"]}-seed{run["seed"]}'
        if sha256(path / "best.pt") != run["checkpoint_sha256"] or sha256(path / "result.json") != run["result_sha256"]:
            raise ValueError("frozen_checkpoint_changed")
    # Opening test content is authorized only after the freeze checks above.
    summary = prepare(archive, config["dataset"], test=True)
    examples = load_bundle(config["dataset"], "test")["test"]
    if not examples:
        raise ValueError("empty_clean_test_split")
    store = frozen_features(
        [e.model_input() for e in examples],
        Path(config["dataset"]) / "clip-test",
        config["weights"],
        config["text_batch_size"],
    )
    tensors = inputs_tensor(examples, store, config)
    vocabulary = ActionVocabulary(experiment["vocabulary"])
    results = []
    for run in freeze["runs"]:
        path = directory / f'{run["model"]}-seed{run["seed"]}'
        checkpoint = torch.load(path / "best.pt", weights_only=True, map_location="cpu")
        model = make_model(config, run["model"], vocabulary)
        model.load_state_dict(checkpoint["state_dict"])
        predictions = predict(model, tensors, vocabulary, config)
        metrics, cases, selected = evaluate(examples, predictions, vocabulary)
        write(path / "test-predictions.json", cases)
        write(path / "test-cases.json", selected)
        result = {
            **run,
            "metrics": metrics,
            "freeze_sha256": sha256(directory / "frozen-checkpoints.json"),
            "test_receipt_sha256": sha256(Path(config["dataset"]) / "test-receipt.json"),
        }
        write(path / "test-result.json", result)
        results.append(result)
    no_op = score_predictions(examples, [[ord("[") + 3, ord("]") + 3, EOS]] * len(examples))[0]
    result = {
        "test_dataset": summary,
        "runs": results,
        "no_op_baseline": no_op,
        "CLIP_input_nearest_training_program_baseline": retrieval_baseline(config, examples, store),
        "test_opened_after_freeze": True,
        "automatic_queue": False,
    }
    write(directory / "test-summary.json", result)
    return result


def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/level2/emnlp2020-task-b-gpu-v3.json"))
    parser.add_argument("--stage", choices=("prepare", "features", "train", "test", "predict"), required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--model", choices=("mlp", "gnn"), default="gnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prediction-output", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("expected_CPU_or_CUDA_device")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA_unavailable_no_silent_CPU_fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    if args.stage == "prepare":
        result = prepare(args.archive, config["dataset"])
    elif args.stage == "features":
        examples = load_bundle(config["dataset"], "trainval")
        store = frozen_features(
            [e.model_input() for rows in examples.values() for e in rows],
            Path(config["dataset"]) / "clip-trainval",
            config["weights"],
            config["text_batch_size"],
        )
        result = {"texts": len(store.keys), "device": "cpu"}
    elif args.stage == "train":
        result = train(config)
    elif args.stage == "test":
        result = final_test(config, args.archive)
    else:
        if args.input is None or args.prediction_output is None:
            raise ValueError("predict_requires_input_and_prediction_output")
        result = infer(config, args.model, args.seed, args.input, args.prediction_output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
