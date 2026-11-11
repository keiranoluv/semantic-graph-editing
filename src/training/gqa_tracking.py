"""W&B epoch logging after durable checkpoints, isolated from training RNG.

Pinned SDK 0.18.7 ignores offline resume. Offline invocations therefore create
distinct snapshots and replay all committed epochs; sync only the latest snapshot.
Online invocations resume one stable run and replay only missing committed epochs.
"""

import random
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.datasets.gqa_full import atomic_write, digest, pin_json


@contextmanager
def preserve_rng():
    cpu, python, numpy = torch.get_rng_state(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        random.setstate(python)
        np.random.set_state(numpy)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class RunTracker:
    def __init__(self, experiment, identity, directory):
        self.directory = Path(directory)
        self.run = None
        self.settings = experiment["config"].get("wandb", {"mode": "disabled"})
        self.mode = self.settings["mode"]
        self.latest_epoch = 0
        if self.mode == "disabled":
            return
        with preserve_rng():
            import wandb

            stable = digest({"identity": identity, "directory": str(self.directory.resolve())})[:24]
            tracking = {
                "logical_id": stable,
                "identity": identity,
                "mode": self.mode,
                "entity": self.settings["entity"],
                "project": self.settings["project"],
            }
            pin_json(self.directory / "wandb-identity.json", tracking)
            physical = stable if self.mode == "online" else f"{stable}-{uuid.uuid4().hex[:8]}"
            local = self.directory / "tracking"
            local.mkdir(exist_ok=True)
            self.run = wandb.init(
                entity=self.settings["entity"],
                project=self.settings["project"],
                id=physical,
                resume="allow" if self.mode == "online" else None,
                mode=self.mode,
                dir=str(local),
                name=f"{experiment['config']['scope']}-{identity['model']}-seed{identity['seed']}",
                group=f"gqa-{experiment['config']['scope']}-{identity['experiment_sha256'][:12]}",
                job_type="training" if experiment["config"]["scope"] == "full" else "integration-smoke",
                config={"experiment": experiment, "run_identity": identity, "metric_units": "fractions"},
                tags=[experiment["config"]["scope"], identity["model"], f"seed-{identity['seed']}"],
                save_code=False,
                settings=wandb.Settings(init_timeout=30, disable_git=True, console="off"),
            )
            self.run.define_metric("epoch")
            for name in ("train/*", "val/*", "best/*", "early_stopping/*"):
                self.run.define_metric(name, step_metric="epoch")
            self.run.summary.update({"logical_id": stable, "execution_status": "running"})
            self.info = {
                **tracking,
                "run_id": physical,
                "url": self.run.url if self.mode == "online" else None,
                "files_dir": self.run.dir,
                "execution_status": "running",
            }
            atomic_write(self.directory / "wandb-latest.json", self.info)

    def replay(self, history):
        if self.run is None:
            return
        with preserve_rng():
            for row in history:
                self.latest_epoch = row["epoch"]
                if row["epoch"] < self.run.step:
                    continue
                values = {
                    "epoch": row["epoch"],
                    **{f"train/{k}": v for k, v in row["train_losses"].items()},
                    **{f"val/{k}": v for k, v in row["validation"].items()},
                    "train/pairs": row["train_pairs"],
                    "train/steps": row["steps"],
                    "train/gradient_norm_max": row["max_gradient_norm_before_clip"],
                    "best/mR@50": row["best_score"],
                    "best/epoch": row["best_epoch"],
                    "early_stopping/bad_epochs": row["early_stopping_bad_epochs"],
                }
                self.run.log(values, step=row["epoch"], commit=True)

    def complete(self, result):
        if self.run is None:
            return
        self.replay(result["history"])
        with preserve_rng():
            final = {
                f"final/{protocol}/{k}": v
                for protocol in ("gc", "ng")
                for k, v in result["validation"][protocol].items()
                if ("R@" in k and isinstance(v, (int, float)))
            }
            self.run.summary.update(
                {
                    **final,
                    "selected_epoch": result["selected_epoch"],
                    "epochs_completed": result["epochs_completed"],
                    "stopping_reason": result["stopping_reason"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "local_best_checkpoint": str(self.directory / "best.pt"),
                    "local_last_checkpoint": str(self.directory / "last.pt"),
                }
            )

    def finish(self, status):
        if self.run is None:
            return
        with preserve_rng():
            self.run.summary["execution_status"] = status
            self.run.finish(exit_code=0 if status == "completed" else 130 if status == "interrupted" else 1)
            atomic_write(self.directory / "wandb-latest.json", {**self.info, "execution_status": status})
