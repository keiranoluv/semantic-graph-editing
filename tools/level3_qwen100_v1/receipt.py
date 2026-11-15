"""Exclude mutable W&B transport files from immutable local evidence receipts."""
from tools.level3_image_graph_v1.core import sha256, write


def finish_run(path, success):
    write(
        path / "receipt.json",
        {
            "schema": "qwen100-run-receipt-v1",
            "execution_complete": success,
            "files_sha256": {
                str(p.relative_to(path)): sha256(p)
                for p in sorted(path.rglob("*"))
                if p.is_file() and "wandb" not in p.relative_to(path).parts
            },
        },
    )
