"""C.1 standalone TransE trainer: python -m scripts.train_transe --help."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.knowledge.transe.core import TransE, check_gate, evaluate, load_kg, negative_sample, set_seed, split_triples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["draft", "reviewed"], default="draft")
    p.add_argument("--review-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    kg = load_kg(args.kg_dir)
    check_gate(args.mode, kg, args.kg_dir, args.review_manifest)
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.dim <= 0:
        p.error("epochs, batch-size, learning rate and dimension must be positive")
    if args.mode == "reviewed" and ("draft" in args.output_dir.parts or "candidate" in args.output_dir.parts):
        p.error("Reviewed outputs must not be mixed into draft/candidate directories")
    if args.output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {args.output_dir}")
    known = {tuple(map(int, row)) for row in kg.triples}
    train, val, test = split_triples(kg.triples, args.seed)
    manifest = {
        "schema_version": 1,
        "milestone": "C.1",
        "mode": args.mode,
        "kg_source": kg.source,
        "kg_dir": str(args.kg_dir.resolve()),
        "input_sha256": kg.hashes,
        "review_manifest_sha256": None,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "lr": args.lr,
        "margin": args.margin,
        "split_policy": "seeded_unique_triple_random_holdout_not_relation_stratified",
        "evaluation": "filtered_head_tail_MRR_Hits@1_3_10",
        "interpretation": "technical_only_not_official"
        if args.mode == "draft"
        else "reviewed_KG_link_prediction_not_PredCls",
    }
    if args.review_manifest:
        from src.knowledge.transe.core import sha256

        manifest["review_manifest_sha256"] = sha256(args.review_manifest)
    if args.dry_run:
        print(
            json.dumps(
                {**manifest, "train_count": len(train), "val_count": len(val), "test_count": len(test)}, indent=2
            )
        )
        return
    set_seed(args.seed)
    device = torch.device(args.device)
    model = TransE(len(kg.entities), len(kg.relations), args.dim, args.margin).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    # Exclusive directory creation; never alter a prior run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        np.savez(args.output_dir / "split_indices.npz", train=train, val=val, test=test)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(train))
            losses = []
            for chunk in np.array_split(order, max(1, (len(order) + args.batch_size - 1) // args.batch_size)):
                pos_np = train[chunk]
                neg_np = negative_sample(pos_np, len(kg.entities), known, rng)
                pos = torch.tensor(pos_np, device=device, dtype=torch.long)
                neg = torch.tensor(neg_np, device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(pos, neg)
                loss.backward()
                optimizer.step()
                model.normalize_entities()
                losses.append(float(loss.detach().cpu()))
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": evaluate(model, val, known, device)}
            history.append(record)
            # Checkpoint per epoch, with optimizer + PRNG state for auditable continuation.
            target = args.output_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "manifest": manifest,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                },
                target,
            )
            print(json.dumps(record))
        result = {"history": history, "test": evaluate(model, test, known, device)}
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Preserve partial run for debugging; NEVER repurpose the directory.
        raise


if __name__ == "__main__":
    main()
