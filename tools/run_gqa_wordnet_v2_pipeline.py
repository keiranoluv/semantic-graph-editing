"""Serial tmux pipeline: v2 mapping/audit -> CUDA TransE -> features -> pilot -> main."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from src.datasets.gqa_full import atomic_write, pin_json
from src.knowledge.transe.core import load_kg, sha256
from tools.gqa_wordnet_v2_features import load_v2
from tools.run_gqa_full import exclusive
from tools.run_gqa_kg_pilot import verify_reference

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("runs/level2/gqa-wordnet-v2-pipeline")
EMBEDDING = Path("runs/level2/gqa-wordnet-transe-v2-bs512")
MAPPING = Path("data/knowledge/gqa-wordnet-v2")
FEATURES = Path("data/knowledge/gqa-wordnet-features-v2")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "embedding", "pilot", "full"), default="all")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    sources = [
        Path(name)
        for name in (
            "tools/run_gqa_wordnet_v2_pipeline.py",
            "tools/prepare_gqa_wordnet_v2.py",
            "tools/gqa_wordnet_v2_features.py",
            "tools/run_gqa_kg_v2.py",
            "scripts/train_transe.py",
        )
    ]
    identity = {
        "schema_version": 1,
        "source_sha256": {str(p): sha256(p) for p in sources},
        "csv_sha256": sha256(Path("reviews/gqa-wordnet-human-v1/submissions/review-ce1dcb634ff05050.csv")),
        "manifest_sha256": sha256(Path("data/processed/level2/gqa-v1/manifest.json")),
        "concurrent_model_trainers": 1,
        "device": "cuda",
        "batch_size": 512,
        "embedding_epochs": 100,
        "embedding_patience": 15,
        "GAT_KG_epochs": 200,
        "GAT_KG_patience": 15,
        "seeds": [42, 123, 3407],
    }
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "WANDB_ENTITY": "keiranoluv-hcmus",
        "WANDB_PROJECT": "semantic-graph-editing",
        "WANDB_RUN_GROUP": "gqa-wordnet-v2",
        "WANDB_TAGS": "gqa-wordnet-v2,batch-512,partial-human-review,sequential,kg-embedding",
    }
    child = {"process": None}
    stopped = {"value": False}

    def stop(signum, _frame):
        stopped["value"] = True
        if child["process"] is not None:
            child["process"].send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def invoke(stage, module, *arguments):
        if stopped["value"]:
            raise SystemExit(130)
        atomic_write(OUTPUT / "status.json", {"status": "running", "stage": stage, "pid": os.getpid()})
        print(f"[STAGE] {stage}: {module}", flush=True)
        child["process"] = subprocess.Popen([sys.executable, "-u", "-m", module, *arguments], env=environment)
        try:
            result = child["process"].wait()
        finally:
            child["process"] = None
        if result:
            atomic_write(
                OUTPUT / "status.json",
                {"status": "interrupted" if stopped["value"] else "failed", "stage": stage, "exit_code": result},
            )
            raise SystemExit(130 if stopped["value"] else result)

    reference = json.loads(Path("runs/level2/gqa-full-v3-bs512/experiment.json").read_text())
    verify_reference(reference)
    with exclusive(OUTPUT):
        pin_json(OUTPUT / "identity.json", identity)
        if args.stage in ("all", "prepare"):
            invoke("prepare", "tools.prepare_gqa_wordnet_v2")
        if args.stage in ("all", "embedding"):
            kg = load_kg(MAPPING / "kg")
            if (EMBEDDING / "metrics.json").exists():
                manifest = json.loads((EMBEDDING / "manifest.json").read_text())
                if manifest["input_sha256"] != kg.hashes or not (EMBEDDING / "best.pt").is_file():
                    raise ValueError("Existing embedding run differs from v2 KG")
                print("[SKIP] completed matching v2 TransE run", flush=True)
            else:
                if EMBEDDING.exists():
                    raise FileExistsError("Incomplete TransE run preserved; use a fresh versioned output for recovery")
                with exclusive(Path("runs/level2/.gqa-kg-training-device")):
                    invoke(
                        "embedding",
                        "scripts.train_transe",
                        "--kg-dir",
                        str(MAPPING / "kg"),
                        "--mode",
                        "draft",
                        "--output-dir",
                        str(EMBEDDING),
                        "--split-policy",
                        "transductive",
                        "--dim",
                        "64",
                        "--epochs",
                        "100",
                        "--patience",
                        "15",
                        "--batch-size",
                        "512",
                        "--lr",
                        "0.001",
                        "--seed",
                        "42",
                        "--device",
                        "cuda",
                        "--wandb-mode",
                        "online",
                    )
            if (FEATURES / "manifest.json").exists():
                _, _, metadata = load_v2(FEATURES, Path("data/processed/level2/gqa-v1/manifest.json"))
                if metadata["checkpoint_sha256"] != sha256(EMBEDDING / "best.pt"):
                    raise ValueError("v2 feature checkpoint changed")
                print("[SKIP] completed verified v2 feature export", flush=True)
            else:
                invoke("features", "tools.gqa_wordnet_v2_features")
        if args.stage in ("all", "pilot"):
            invoke("pilot", "tools.run_gqa_kg_v2", "--stage", "pilot")
        if args.stage in ("all", "full"):
            invoke("full", "tools.run_gqa_kg_v2", "--stage", "full")
        verify_reference(reference)
        atomic_write(
            OUTPUT / "status.json",
            {"status": "completed", "requested_stage": args.stage, "test_used": False, "concurrent_model_trainers": 1},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
