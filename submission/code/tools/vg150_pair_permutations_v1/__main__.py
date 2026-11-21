"""Six matched pair-fusion controls on VG150. Run stages from the repository root."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import torch

from src import level1_fusion_matrix as old
from src.datasets.gqa_full import atomic_write
from src.knowledge.transe.core import sha256
from tools.level1_reference_test27_v1 import validate
from tools.run_gqa_full import exclusive

ROOT = old.ROOT
REV = "3d51cbf4e2276c3d0e7e6994ea5bec79c3219d6b"
NAME = "vg150-pair-permutations-v1"
RUNS = ROOT / "runs" / NAME
REPORT = ROOT / "reports/level1" / NAME
CONTROL = ROOT / "reports/level1/reference-loss-fusion-test27-v1"
PERMUTATIONS = {
    20261025: "0564d65f33345ecf482a540ef1423b28b69bdc172526e242050c1218a251e9b3",
    20261026: "42e655b5eed6a1c0aca0eedb0915c1446c0e3e08d4bb0e7a4ee45d092a275fc8",
    20261027: "afa96921900b21d0450027239d960d8fe7b4e793e8ad7df5addea298270ef658",
}
NEW = (20261026, 20261027)
JOBS = [(p, s) for p in NEW for s in old.SEEDS]


def read(path):
    return json.loads(path.read_text())


def controls():
    summary = CONTROL / "summary.json"
    if sha256(summary) != "cc0dad86b71403b7b9302f7bea522bcd4b1d70d8506130de64c83185ae4a68a2":
        raise ValueError("Historical test summary changed")
    hashes = {r["path"]: r["sha256"] for r in read(summary)["reports"]}
    rows = {}
    for kind in ("real", "shuffled"):
        for seed in old.SEEDS:
            path = CONTROL / f"fusion_pair_{kind}-seed{seed}-test26446.json"
            if sha256(path) != hashes[str(path.relative_to(ROOT))]:
                raise ValueError(f"Historical control changed: {path}")
            row = read(path)
            if any(sha256(ROOT / p) != h for p, h in row["evaluator_sha256"].items()):
                raise ValueError("Historical evaluator changed")
            rows[kind, seed] = row
    return rows


def expected_config(controls, perm, seed):
    return {**controls["shuffled", seed]["config"], "shuffle_seed": perm}


def directory(perm, seed):
    return RUNS / f"pair-shuffled-{perm}-seed{seed}"


def command(perm, seed, dest, **overrides):
    cmd = old.train_command("pair_shuffled", seed, "online")
    for key, value in {"shuffle-seed": perm, "output-dir": dest, **overrides}.items():
        cmd[cmd.index("--" + key) + 1] = str(value)
    return cmd


def verify_permutation(state, perm):
    indices = state["kg_class_permutation"].tolist()
    if hashlib.sha256(bytes(indices)).hexdigest() != PERMUTATIONS[perm]:
        raise ValueError(f"Incorrect KG mapping for {perm}")


def completed(dest, config):
    if not (dest / "last.pt").exists():
        return False
    last = torch.load(dest / "last.pt", map_location="cpu", weights_only=True)
    if last["config"] != config or last["git_sha"] != REV or not last["wandb_id"]:
        raise ValueError(f"Different training run: {dest}")
    verify_permutation(last["model"], config["shuffle_seed"])
    if last["epoch"] < config["max_epochs"] and last["bad_epochs"] < config["patience"]:
        return False
    best = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)
    peak = max(last["history"], key=lambda r: r["val/mR@50"])
    if (
        best["config"] != config
        or best["epoch"] != peak["epoch"]
        or best["git_sha"] != REV
        or best["best_score"] != peak["val/mR@50"]
        or best["wandb_id"] != last["wandb_id"]
    ):
        raise ValueError(f"Incorrect validation-selected checkpoint: {dest}")
    verify_permutation(best["model"], config["shuffle_seed"])
    return True


def training_source(tmp, cfg):
    # Use the exact trainer, including its validation code and source-tree hashes.
    src = tmp / "historical"
    objects = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-path", "objects"], text=True
    ).strip()
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    (src / ".git/objects/info/alternates").write_text(objects + "\n")
    subprocess.run(["git", "-C", str(src), "update-ref", "--no-deref", "HEAD", REV], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", REV, "--", "src", "scripts"], check=True)
    hashes = {str(p.relative_to(src)): sha256(p) for d in ("src", "scripts") for p in (src / d).rglob("*.py")}
    if hashes != cfg["code_sha256"]:
        raise ValueError("Historical training source differs from the original control")
    return src


def train(rows, tmp, smoke):
    versions = read(old.run_dir("pair_shuffled", 42) / "environment.json")["packages"]
    if any(version(k) != v for k, v in {**versions, "wandb": "0.18.7"}.items()):
        raise ValueError("Use the torch 2.7.0 GQA training environment")
    src = training_source(tmp, rows["shuffled", 42]["config"])
    cfg = rows["shuffled", 42]["config"]
    for artifact, key in [
        (old.FEATURES / "manifest.json", "kg_manifest_sha256"),
        (old.RELATIONS / "manifest.json", "kg_relation_manifest_sha256"),
        (old.PRIOR, "loss_prior_sha256"),
        (old.CACHE / "manifest.json", "visual_cache_manifest_sha256"),
    ]:
        if sha256(artifact) != cfg[key]:
            raise ValueError(f"Different training input: {artifact}")
    env = {**os.environ, "PYTHONPATH": str(src), "WANDB_DIR": str(RUNS)}
    if smoke:
        from scripts.prepare_pair_kg_prior import make_prior

        prior = tmp / "prior.json"
        atomic_write(prior, make_prior(ROOT / "data/raw/vg150", 8, 3))
    for i, (perm, seed) in enumerate(JOBS, 1):
        dest = directory(perm, seed)
        config = expected_config(rows, perm, seed)
        if smoke:
            if seed != old.SEEDS[0]:
                continue
            dest = tmp / dest.name
            cmd = command(
                perm,
                seed,
                dest,
                **{
                    "device": "cpu",
                    "epochs": 1,
                    "max-train-images": 8,
                    "max-val-images": 8,
                    "loss-prior": prior,
                    "wandb-mode": "disabled",
                },
            )
        else:
            if completed(dest, config):
                print(f"[SKIP {i}/6] {dest.name}", flush=True)
                continue
            cmd = command(perm, seed, dest)
            if (dest / "last.pt").exists():
                cmd += ["--resume", str(dest / "last.pt")]
            elif dest.exists():
                # Preserve partial startup output (e.g. interruption before epoch 1).
                backup = Path(tempfile.mkdtemp(prefix=dest.name + "-interrupted-", dir=RUNS))
                dest.rename(backup / dest.name)
        print(f"[RUN {i}/6] {dest.name}", flush=True)
        subprocess.run(cmd, cwd=src, env=env, check=True)
        if smoke:
            best = torch.load(dest / "best.pt", map_location="cpu", weights_only=True)
            verify_permutation(best["model"], perm)
            for k, v in config.items():
                if k not in ("train_limit", "val_limit", "max_epochs", "loss_prior_sha256") and best["config"][k] != v:
                    raise ValueError(f"Smoke config mismatch: {k}")
            report = tmp / f"{perm}-val.json"
            score(dest / "best.pt", report, smoke=True)
            row = read(report)
            if (
                row["images_evaluated"] != 8
                or row["split"] != "val"
                or row["config"] != best["config"]
                or row["evaluator_sha256"] != rows["shuffled", seed]["evaluator_sha256"]
                or row["checkpoint_sha256"] != sha256(dest / "best.pt")
            ):
                raise ValueError("CPU reference evaluation mismatch")
            print(f"[CPU PASS] permutation={perm} train=8 val=8", flush=True)
        elif not completed(dest, config):
            raise ValueError(f"Incomplete run: {dest}")


def score(cp, output, smoke=False):
    cmd = [
        str(ROOT / "myenv/bin/python"),
        "-u",
        "-m",
        "scripts.evaluate_level1",
        "--checkpoint",
        str(cp),
        "--protocol",
        "reference-v1",
        "--batch-size",
        "128",
    ]
    cmd += (
        ["--split", "val", "--device", "cpu", "--max-images", "8"]
        if smoke
        else ["--split", "test", "--allow-test", "--device", "cuda"]
    )
    # Publish only complete JSON so interruption can be retried.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        result = Path(tmp) / "report.json"
        # The frozen evaluator imports wandb via train_mlp. Give it the compatible
        # protobuf from the training venv, while keeping its original torch/numpy.
        packages = ROOT / "data/cache/gqa-gat-loss-torch270/venv/lib/python3.12/site-packages"
        (Path(tmp) / "google").symlink_to(packages / "google", target_is_directory=True)
        subprocess.run(cmd + ["--output", str(result)], cwd=ROOT, env={**os.environ, "PYTHONPATH": tmp}, check=True)
        result.replace(output)


def test(rows):
    base = rows["shuffled", 42]
    if version("torch") != base["torch_version"] or version("numpy") != base["numpy_version"]:
        raise ValueError("Use the original myenv test runtime")
    for perm, seed in JOBS:
        if not completed(directory(perm, seed), expected_config(rows, perm, seed)):
            raise ValueError(f"Training incomplete: permutation={perm} seed={seed}")
    shuffled = {(old.SHUFFLE_SEED, s): rows["shuffled", s] for s in old.SEEDS}
    for i, (perm, seed) in enumerate(JOBS, 1):
        dest = directory(perm, seed)
        cp = dest / "best.pt"
        path = REPORT / f"{dest.name}-test26446.json"
        print(f"[{'REUSE' if path.exists() else 'TEST'} {i}/6] {path.name}", flush=True)
        if not path.exists():
            score(cp, path)
        row = read(path)
        validate(row, {"seed": seed, "checkpoint_sha256": sha256(cp)}, base["config"])
        if (
            row["config"] != expected_config(rows, perm, seed)
            or row["evaluation_batch_size"] != 128
            or row["torch_version"] != base["torch_version"]
            or row["evaluator_sha256"] != base["evaluator_sha256"]
        ):
            raise ValueError(f"Test protocol mismatch: {path}")
        shuffled[perm, seed] = row
    comparisons = []
    for perm in PERMUTATIONS:
        per_seed = {}
        for seed in old.SEEDS:
            real, control = rows["real", seed], shuffled[perm, seed]
            per_seed[seed] = {
                k: {
                    "real_pct": 100 * real[k],
                    "shuffled_pct": 100 * control[k],
                    "real_minus_shuffled_pp": 100 * (real[k] - control[k]),
                }
                for k in ("mR@50", "R@50")
            }
        delta = {k: mean(r[k]["real_minus_shuffled_pp"] for r in per_seed.values()) for k in ("mR@50", "R@50")}
        comparisons.append({"permutation": perm, "per_seed": per_seed, "mean_real_minus_shuffled_pp": delta})
        print(f"[PAIRED] {perm}: {delta}", flush=True)
    atomic_write(
        REPORT / "summary.json",
        {
            "protocol": base["protocol"],
            "test_images": 26446,
            "scope": "exploratory follow-up on previously evaluated test split",
            "comparisons": comparisons,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "test"))
    stage = parser.parse_args().stage
    rows = controls()
    if stage == "test":
        with exclusive(REPORT):
            test(rows)
    elif stage == "smoke":
        with tempfile.TemporaryDirectory() as tmp:
            train(rows, Path(tmp), True)
    else:
        with exclusive(RUNS), tempfile.TemporaryDirectory() as tmp:
            train(rows, Path(tmp), False)


if __name__ == "__main__":
    main()
