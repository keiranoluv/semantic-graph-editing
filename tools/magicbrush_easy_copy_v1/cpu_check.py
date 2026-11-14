"""Real-data CPU smoke with actual frozen CLIP embeddings; never starts CUDA."""
import json

import torch

from tools.edit_model_v1.text import TextStore, text_key
from tools.emnlp2020_graph_modification_v1.dataset import sha256, write
from tools.magicbrush_experiment_v3.cache import input_texts
from tools.magicbrush_experiment_v3.vocabulary import prototype_inputs, prototype_vectors
from tools.prepare_magicbrush import ROOT, digest

from .features import GroundingStore, cache_inputs
from .model import StructuredPredictor
from .runner import preflight
from .runtime import targets_for, training_step, validate


def main():
    torch.set_num_threads(2)
    cfg, examples, identity, _ = preflight()
    vocab = identity["vocabulary"]
    for items in examples.values():
        for start in range(0, len(items), 16):
            targets_for(items[start : start + 16], vocab, cfg, use_copy=True)
    sample = examples["train"][:16]
    texts = input_texts(sample + cache_inputs(sample) + prototype_inputs(vocab))
    import clip

    encoder, _ = clip.load(str(ROOT / cfg["weights"]), device="cpu", jit=False)
    encoder.eval().requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            tokens = clip.tokenize(texts[start : start + 32], truncate=True)
            chunks.append(torch.nn.functional.normalize(encoder.encode_text(tokens).float(), dim=-1))
            if start % 256 == 0:
                print(json.dumps({"stage": "cpu_clip", "texts": start, "total": len(texts)}), flush=True)
    assert all(not p.requires_grad for p in encoder.parameters())
    del encoder
    vectors = torch.cat(chunks)
    store = GroundingStore(TextStore([text_key(t) for t in texts], vectors), vocab)
    checks = []
    for use_copy in (False, True):
        for kind in ("mlp", "gnn"):
            torch.manual_seed(42)
            model = StructuredPredictor(
                kind=kind,
                use_copy=use_copy,
                vocabulary=vocab,
                prototypes=prototype_vectors(store, vocab),
                **cfg["model"],
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            loss, count = training_step(model, optimizer, sample, store, cfg, "cpu")
            metrics, rows = validate(model, sample, store, cfg, "cpu")
            assert len(rows) == count == 16
            checks.append(
                {
                    "model": kind,
                    "copy": use_copy,
                    "finite_train_loss": loss,
                    "decoded_rows": len(rows),
                    "copy_bias_gradient": model.copy_bias.grad.tolist() if model.copy_bias.grad is not None else None,
                }
            )
            print(json.dumps({"stage": "cpu_graph_smoke", **checks[-1]}), flush=True)
    report = {
        "experiment_sha256": digest(identity),
        "release_lock_sha256": sha256(ROOT / "configs/level2/magicbrush-easy-copy-v1.lock.json"),
        "scope": "16 real training pairs, actual frozen CLIP on CPU, one update per arm; NOT a performance experiment",
        "all_trainval_targets_encoded": sum(map(len, examples.values())),
        "sample_ids": [e.id for e in sample],
        "clip_texts": len(texts),
        "encoder_frozen": True,
        "device": "cpu",
        "gpu_jobs_started": False,
        "test_used": False,
        "checks": checks,
    }
    write(ROOT / cfg["reports"] / "cpu-smoke.json", report)


if __name__ == "__main__":
    main()
