"""C.1 reviewed ConceptNet ComplEx with filtered logistic negatives."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.knowledge.complex import ComplexReviewed, evaluate_complex
from src.knowledge.transe.core import (
    check_gate,
    load_kg,
    negative_sample,
    set_seed,
    split_triples,
    split_triples_transductive,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kg-dir", type=Path, required=True)
    p.add_argument("--mode", choices=["draft", "reviewed"], default="draft")
    p.add_argument("--review-manifest", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-policy", choices=["random", "transductive"], default="random")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument(
        "--eval-max-triples",
        type=int,
        default=None,
        help="Deterministic validation/test subset size; omitted means full evaluation",
    )
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=None, help="Stop after this many epochs without val MRR gain")
    p.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation MRR gain")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--negative-count", type=int, default=32)
    p.add_argument("--l2-regularization", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    kg = load_kg(args.kg_dir)
    check_gate(args.mode, kg, args.kg_dir, args.review_manifest)
    if kg.source != "conceptnet" or args.mode != "reviewed":
        p.error("This RotatE experiment requires a reviewed ConceptNet KG")
    if not (
        math.isfinite(args.val_ratio)
        and math.isfinite(args.test_ratio)
        and 0 < args.val_ratio < 1
        and 0 < args.test_ratio < 1
        and args.val_ratio + args.test_ratio < 1
    ):
        p.error("val-ratio and test-ratio must be finite positive fractions summing to less than one")
    if args.eval_max_triples is not None and args.eval_max_triples <= 0:
        p.error("--eval-max-triples must be positive")
    if args.patience is not None and args.patience <= 0:
        p.error("--patience must be positive")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        p.error("--min-delta must be finite and nonnegative")
    if not (math.isfinite(args.lr) and args.lr > 0):
        p.error("lr must be finite and positive")
    if args.negative_count <= 0 or not math.isfinite(args.l2_regularization) or args.l2_regularization < 0:
        p.error("negative-count must be positive and l2-regularization finite and nonnegative")
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0 or args.dim <= 0 or args.dim % 2:
        p.error("epochs, batch-size and learning rate must be positive; --dim must be positive and even")
    if args.mode == "reviewed" and ("draft" in args.output_dir.parts or "candidate" in args.output_dir.parts):
        p.error("Reviewed outputs must not be mixed into draft/candidate directories")
    if args.output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {args.output_dir}")
    splitter = split_triples_transductive if args.split_policy == "transductive" else split_triples
    train, val, test = splitter(kg.triples, args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    train_known = {tuple(map(int, row)) for row in train}
    val_known = train_known | {tuple(map(int, row)) for row in val}
    test_known = val_known | {tuple(map(int, row)) for row in test}

    # Subset evaluation ONLY; training split and the filtered-positive set stay unchanged.
    def eval_subset(data, offset):
        if args.eval_max_triples is None or args.eval_max_triples >= len(data):
            return data
        rng = np.random.default_rng(args.seed + offset)
        selected = np.sort(rng.choice(len(data), size=args.eval_max_triples, replace=False))
        return data[selected]

    val_eval = eval_subset(val, 1001)
    test_eval = eval_subset(test, 2001)
    manifest = {
        "schema_version": 1,
        "milestone": "C.1",
        "embedding_model": "complex",
        "embedding_version": 1,
        "score_function": "real_hermitian_trilinear_product",
        "mode": args.mode,
        "kg_source": kg.source,
        "kg_dir": str(args.kg_dir.resolve()),
        "input_sha256": kg.hashes,
        "review_manifest_sha256": None,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "checkpoint_selection": "best_validation_MRR" if args.patience is not None else "final_epoch_legacy",
        "wandb_mode": args.wandb_mode,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "eval_max_triples": args.eval_max_triples,
        "eval_subset": args.eval_max_triples is not None and (len(val_eval) < len(val) or len(test_eval) < len(test)),
        "eval_val_triples": len(val_eval),
        "eval_test_triples": len(test_eval),
        "eval_sampling": "seeded_without_replacement_separate_validation_test"
        if args.eval_max_triples is not None
        else "all_triples",
        "batch_size": args.batch_size,
        "dim": args.dim,
        "complex_dim": args.dim // 2,
        "entity_export": "real_imag_concat_l2_normalized",
        "loss": "filtered_logistic_l2",
        "negative_count": args.negative_count,
        "l2_regularization": args.l2_regularization,
        "lr": args.lr,
        "split_policy": (
            "seeded_transductive_coverage_holdout"
            if args.split_policy == "transductive"
            else "seeded_unique_triple_random_holdout_not_relation_stratified"
        ),
        "evaluation": "filtered_head_tail_MRR_Hits@1_3_10",
        "negative_filter": "train_only",
        "validation_rank_filter": "train_and_validation_only",
        "test_rank_filter": "train_validation_and_test",
        "interpretation": "technical_only_not_official"
        if args.mode == "draft"
        else "reviewed_KG_link_prediction_not_PredCls",
    }
    if args.review_manifest:
        from src.knowledge.transe.core import sha256

        manifest["review_manifest_sha256"] = sha256(args.review_manifest)
        review_info = json.loads(args.review_manifest.read_text(encoding="utf-8"))
        manifest["review_status"] = review_info.get("review_status")
        manifest["review_basis"] = review_info.get("review_basis")
        manifest["per_class_rationale_provided"] = review_info.get("per_class_rationale_provided")
    if args.dry_run:
        print(
            json.dumps(
                {**manifest, "train_count": len(train), "val_count": len(val), "test_count": len(test)}, indent=2
            )
        )
        return
    set_seed(args.seed)
    device = torch.device(args.device)
    model = ComplexReviewed(len(kg.entities), len(kg.relations), args.dim // 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    # Exclusive directory creation; never alter a prior run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run = None
    try:
        np.savez(args.output_dir / "split_indices.npz", train=train, val=val, test=test)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project="semantic-graph-editing",
                name=args.output_dir.name,
                mode=args.wandb_mode,
                config=manifest,
            )
        history = []
        best_mrr, best_epoch, bad_epochs = -math.inf, 0, 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(train))
            losses = []
            for chunk in np.array_split(order, max(1, (len(order) + args.batch_size - 1) // args.batch_size)):
                pos_np = train[chunk]
                neg_np = np.stack(
                    [negative_sample(pos_np, len(kg.entities), train_known, rng) for _ in range(args.negative_count)],
                    axis=1,
                )
                pos = torch.tensor(pos_np, device=device, dtype=torch.long)
                neg = torch.tensor(neg_np, device=device, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss = model.filtered_loss(pos, neg, args.l2_regularization)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            record = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val": evaluate_complex(model, val_eval, val_known, device),
            }
            score = record["val"]["MRR"]
            if not math.isfinite(score):
                raise RuntimeError("Cannot select ComplEx checkpoint without finite validation MRR")
            improved = score > best_mrr + args.min_delta
            if improved:
                best_mrr, best_epoch, bad_epochs = score, epoch, 0
            else:
                bad_epochs += 1
            history.append(record)
            # Checkpoint per epoch, with optimizer + PRNG state for auditable continuation.
            target = args.output_dir / f"epoch_{epoch:04d}.pt"
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "manifest": manifest,
                "numpy_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
            }
            torch.save(checkpoint, target)
            if improved:
                torch.save(checkpoint, args.output_dir / "best.pt")
            print(json.dumps(record))
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "train/loss": record["train_loss"],
                        **{f"val/{name}": value for name, value in record["val"].items()},
                        "val/best_MRR": best_mrr,
                        "val/bad_epochs": bad_epochs,
                    },
                    step=epoch,
                )
            if args.patience is not None and bad_epochs >= args.patience:
                print(f"[EARLY STOP] best_epoch={best_epoch} val_MRR={best_mrr:.6f}", flush=True)
                break
        selected_epoch = best_epoch if args.patience is not None else epoch
        if args.patience is not None:
            best = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=True)
            model.load_state_dict(best["model"])
        test_metrics = evaluate_complex(model, test_eval, test_known, device)
        result = {
            "history": history,
            "test": test_metrics,
            "selection": {
                "metric": "validation_filtered_MRR",
                "best_epoch": best_epoch,
                "best_val_MRR": best_mrr,
                "selected_epoch": selected_epoch,
                "early_stopped": len(history) < args.epochs,
            },
            "evaluation_protocol": {
                "subset": manifest["eval_subset"],
                "max_triples": args.eval_max_triples,
                "val_evaluated": len(val_eval),
                "test_evaluated": len(test_eval),
                "val_total": len(val),
                "test_total": len(test),
                "sampling_seed": args.seed,
                "official": False if manifest["eval_subset"] else args.mode == "reviewed",
            },
        }
        (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if run is not None:
            run.summary["best_epoch"] = best_epoch
            run.summary["best_val_MRR"] = best_mrr
            run.summary["selected_epoch"] = selected_epoch
            for name, value in test_metrics.items():
                run.summary[f"test/{name}"] = value
    finally:
        # Preserve partial run for debugging; NEVER repurpose the directory.
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
