"""CPU contracts and a synthetic pipeline: no GPU, model weights or W&B network calls."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import ROOT, read, sha256
from tools.level3_qwen21_v1.core import bundle
from tools.level3_qwen200_v1 import core, runner
from tools.level3_qwen200_v1.viewer import render


def packs():
    return (
        read(core.MANIFEST),
        read(ROOT / "configs/level3/qwen100-instructions-v1.json"),
        read(core.COHORT),
    )


def test_frozen_200_requests_are_two_independent_edits_per_source():
    pack, parent, cohort = packs()
    assert len(core.validate_extension(pack, parent, cohort)) == 100
    assert cohort["tasks"][:100] == parent["tasks"]
    assert sum(r["operation"]["kind"] == "attribute" for r in pack["tasks"]) == 10
    assert all(r["human_outcome_label"] is None for r in pack["tasks"])


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda rows, old: rows.pop(), "exact_100"),
        (lambda rows, old: rows[0].update(task_id=old[0]["task_id"]), "reuse_old_task_ids"),
        (lambda rows, old: rows[1].update(task_id=rows[0]["task_id"]), "duplicate_extension_task_id"),
        (lambda rows, old: rows[0].update(source_path="runs/old/target.png"), "original_source_not_target"),
        (lambda rows, old: rows[0].update(instruction_en=old[0]["instruction_en"]), "duplicate_new_instruction"),
        (lambda rows, old: rows[0].update(instruction_en=rows[1]["instruction_en"]), "duplicate_new_instruction"),
        (lambda rows, old: rows[0].update(source_revision_sha="bad"), "original_source_not_target"),
        (lambda rows, old: rows[0].update(human_outcome_label=True), "forge_human_outcome"),
        (lambda rows, old: rows[0].update(visual_evidence=""), "visual_authoring"),
        (lambda rows, old: rows[0].update(request_index=1), "pair_identity"),
        (lambda rows, old: rows[0].update(input_policy="edit_previous_target"), "sequential_edits"),
        (lambda rows, old: rows[8].update(sgg_edit_fulfillment_support="supported"), "support_must_be_explicit"),
        (lambda rows, old: rows[8]["operation"].update(target_value="yellow"), "color_change_required"),
        (lambda rows, old: rows[8]["operation"].update(attribute="material"), "color_change_required"),
        (lambda rows, old: rows[0]["operation"].update(kind="attribute"), "operation_mix"),
        (lambda rows, old: rows[1]["operation"].update(replacement_label="motorcycle"), "change_category"),
    ],
)
def test_invalid_or_leaky_extension_rejected(mutation, expected):
    pack, parent, cohort = packs()
    mutation(pack["tasks"], parent["tasks"])
    cohort["tasks"] = parent["tasks"] + pack["tasks"]
    with pytest.raises(ValueError, match=expected):
        core.validate_extension(pack, parent, cohort)


def test_combined_manifest_cannot_rewrite_old_instruction():
    pack, parent, cohort = packs()
    cohort["tasks"][0]["instruction_en"] = "Rewritten old request"
    with pytest.raises(ValueError, match="preserve_old_100_exactly"):
        core.validate_extension(pack, parent, cohort)


def test_output_refuses_overwrite_and_old_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "RUNS", tmp_path / "extension")
    p = core.new_output(tmp_path / "extension/new")
    with pytest.raises(FileExistsError):
        core.new_output(p)
    with pytest.raises(ValueError, match="outside_extension"):
        core.new_output(tmp_path / "old100/new")
    with pytest.raises(ValueError, match="not_old_100"):
        core.resume_cases(tmp_path / "old100/run", {}, [])


def test_commit_gate_before_cuda(monkeypatch, tmp_path):
    def dirty():
        raise ValueError("commit_extension_code_before_GPU_inference")

    monkeypatch.setattr(runner, "identity", dirty)
    monkeypatch.setattr(runner, "runtime", lambda: pytest.fail("must not touch CUDA before commit gate"))
    with pytest.raises(ValueError, match="commit_extension"):
        runner.run(NS(output=tmp_path, resume_from=None))


@pytest.fixture
def synthetic_runtime(tmp_path, monkeypatch):
    """Run real orchestration/receipts on CPU with a deliberately fake editor."""
    cfg, tasks = read(core.CONFIG), packs()[0]["tasks"]
    image = tmp_path / "synthetic.png"
    im = Image.new("RGB", (64, 64), "white")
    im.putpixel((0, 0), (0, 0, 0))
    im.save(image)
    for t in tasks:
        t["source_path"] = str(image)
    state = NS(calls=[], fail_at=None, short_steps=False)

    class Generator:
        def __init__(self, device):
            assert device == "cuda"

        def manual_seed(self, value):
            assert value == 42
            return self

    class FakeOOM(Exception):
        pass

    class Part:
        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False

        def enable_tiling(self):
            pass

    class Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"] and kwargs["torch_dtype"] == "synthetic_bf16"
            p = cls()
            p.text_encoder = p.transformer = p.vae = Part()
            p.scheduler = NS(config={"synthetic": True})
            return p

        def enable_model_cpu_offload(self, gpu_id):
            assert gpu_id == 0

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            assert kwargs["num_inference_steps"] == 40 and kwargs["num_images_per_prompt"] == 1
            state.calls.append(kwargs["prompt"])
            if len(state.calls) == state.fail_at:
                raise FakeOOM("synthetic OOM; no GPU was used")
            for i in range(39 if state.short_steps else 40):
                kwargs["callback_on_step_end"](self, i, None, {})
            return NS(images=[im.copy()])

    class Monitor:
        samples = []

        def start(self):
            pass

        def close(self):
            pass

        def summary(self, *args):
            return {"synthetic": True}

    torch = NS(
        bfloat16="synthetic_bf16",
        Generator=Generator,
        inference_mode=nullcontext,
        OutOfMemoryError=FakeOOM,
        cuda=NS(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            get_device_name=lambda: "synthetic CPU test",
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        ),
        backends=NS(cuda=NS(matmul=NS()), cudnn=NS()),
    )
    tracking = NS(id="synthetic", url="synthetic://no-network", log=lambda _: None, finish=lambda **_: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "wandb", NS(init=lambda **_: tracking))
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setattr(core, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "identity", lambda: {"commit": "synthetic", "files_sha256": {"synthetic": "fixed"}})
    monkeypatch.setattr(runner, "preflight", lambda: (cfg, tasks))
    monkeypatch.setattr(runner, "runtime", lambda: Pipeline)
    monkeypatch.setattr(runner, "MemoryMonitor", Monitor)
    monkeypatch.setattr(runner.importlib.metadata, "distributions", lambda: [])
    return state, tasks, tmp_path / "runs"


def test_only_100_new_tasks_and_no_extra_queue(synthetic_runtime):
    state, tasks, root = synthetic_runtime
    assert runner.run(NS(output=root / "first", resume_from=None)) == 0
    assert state.calls == [t["instruction_en"] for t in tasks]
    rows = bundle(root / "first", "results.json")
    assert len(rows) == 100 and all(r["completed_steps"] == 40 and r["status"] == "ok" for r in rows)
    review = read(root / "first/review-template.json")
    assert all(r["edit_fulfillment"] is None and r["preservation"] is None for r in review["reviews"])


def test_interruption_and_resume_preserve_successes_without_rerunning_them(synthetic_runtime):
    state, tasks, root = synthetic_runtime
    state.fail_at = 3
    assert runner.run(NS(output=root / "interrupted", resume_from=None)) == 2
    assert read(root / "interrupted/failure.json")["oom"] is True
    receipt_hash = sha256(root / "interrupted/receipt.json")
    first = read(root / "interrupted/results.json")
    assert [r["status"] for r in first[:4]] == ["ok", "ok", "error", "not_run"]
    assert read(root / "interrupted/receipt.json")["execution_complete"] is False
    state.calls.clear()
    state.fail_at = None
    assert runner.run(NS(output=root / "resumed", resume_from=root / "interrupted")) == 0
    assert state.calls == [t["instruction_en"] for t in tasks[2:]]
    resumed = bundle(root / "resumed", "results.json")
    assert resumed[:2] == first[:2] and len(resumed) == 100
    assert sha256(root / "interrupted/receipt.json") == receipt_hash


def test_incomplete_diffusion_steps_do_not_become_success(synthetic_runtime):
    state, _, root = synthetic_runtime
    state.short_steps = True
    assert runner.run(NS(output=root / "short", resume_from=None)) == 2
    assert len(state.calls) == 1
    assert read(root / "short/execution.json")["successful"] == 0


def test_review_two_fields_escaped_prompt_and_missing_target(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "green").save(source)
    render(
        [
            {
                "task_id": "synthetic",
                "source_path": str(source),
                "target_path": str(source),
                "prompt": "<script>alert(1)</script>",
                "status": "error",
            }
        ],
        tmp_path,
    )
    page = (tmp_path / "comparison.html").read_text()
    assert "<script>alert(1)</script>" not in page
    assert '"target": null' in page and "textarea" not in page and "Pix2Pix" not in page
    assert read(tmp_path / "review-template.json")["reviews"][0]["edit_fulfillment"] is None
