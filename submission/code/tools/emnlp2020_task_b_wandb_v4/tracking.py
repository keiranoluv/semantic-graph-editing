"""Online tracking for main runs; smoke tests never import or initialize W&B."""

import os

from tools.emnlp2020_graph_modification_v1.dataset import digest


def flatten(metrics, prefix):
    result = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            result[name] = value
    return result


def start(config, run_identity):
    if config["execution"] == "smoke":
        return None
    if config["execution"] != "main" or config["wandb"]["mode"] != "online":
        raise ValueError("main_training_requires_online_WandB")
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("WANDB_MODE_must_be_online_for_main_training")
    import wandb

    identifier = digest(run_identity)[:20]
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity") or os.environ.get("WANDB_ENTITY"),
        group=f'{config["schema"]}-{run_identity["experiment_identity"][:12]}',
        name=f'{config["schema"]}-{run_identity["model"]}-seed{run_identity["seed"]}',
        id=identifier,
        resume="allow",
        mode="online",
        force=True,
        dir=config["output"],
        save_code=False,
        settings=wandb.Settings(disable_git=True),
        config={"recipe": config, **run_identity},
        tags=["emnlp2020", "crowdsourced", "graph-editing", "v4", run_identity["model"]],
    )
    if run is None or run.settings.mode != "online":
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError("online_WandB_not_initialized_no_training_allowed")
    run.define_metric("epoch")
    for prefix in ("train/*", "val/*", "early_stop/*", "system/*"):
        run.define_metric(prefix, step_metric="epoch")
    return run


def log_history(run, history):
    if run is None:
        return
    # last.pt is durable before log: replay locally saved epochs not yet logged.
    next_step = run.step
    for row in history:
        if row["epoch"] >= next_step:
            run.log(row, step=row["epoch"])
