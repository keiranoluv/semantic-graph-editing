import csv
import json

import pytest
from PIL import Image

from tools.level3_mllm_judge_v1 import backend, protocol, report, runner

RAW = '{"fulfillment":2,"preservation":3,"evidence":"The roof is blue and the wall remains intact."}'
MEMORY = {"allocated_mib": 10, "reserved_mib": 11}


@pytest.mark.parametrize(
    "raw",
    [
        RAW.replace("2", "true"),
        RAW.replace("2", "4.0"),
        RAW.replace("2", "5"),
        RAW.replace("2", "-1"),
        RAW.replace("2,", '2,"fulfillment":3,'),
        RAW + " done",
        "```json\n" + RAW + "\n```",
        RAW.replace("The roof", "One. Two roofs"),
        "{}",
        RAW.replace('"The roof is blue and the wall remains intact."', '""'),
    ],
)
def test_invalid_output_abstains(raw):
    result = protocol.parse(raw)
    assert result["status"] == "invalid_format"
    assert all(result[k] is None for k in ("fulfillment", "preservation", "J", "evidence"))


def test_product():
    for f in range(5):
        for p in range(5):
            raw = json.dumps(dict(fulfillment=f, preservation=p, evidence="The car changed color."))
            assert protocol.parse(raw)["J"] == f * p / 16


def test_true_processor_keeps_whole_frames_and_order(tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoProcessor

    cfg = protocol.read(protocol.CONFIG)
    try:
        folder = protocol.snapshot(cfg, weights=False)
    except (FileNotFoundError, LocalEntryNotFoundError):
        pytest.skip("Pinned processor files unavailable offline")
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, use_fast=True)
    paths = [tmp_path / "red.png", tmp_path / "blue.png"]
    for path, color, size in zip(paths, ["red", "blue"], [(1600, 777), (333, 999)], strict=True):
        Image.new("RGB", size, color).save(path)
    case = dict(source=paths[0], target=paths[1], instruction="Change {the bird}.", secret="hidden label")
    prompt = protocol.checked(cfg["prompt"]).read_text()
    chat = processor.apply_chat_template(
        protocol.messages(prompt, case["instruction"]), tokenize=False, add_generation_prompt=True
    )
    assert chat.count("<|image_pad|>") == 2 and chat.rfind("<|image_pad|>") < chat.index(case["instruction"])
    assert "hidden label" not in chat
    batch, sizes = backend.encode(processor, prompt, cfg, case)
    assert [s["content_wh"] for s in sizes] == [[1024, 497], [333, 999]]
    assert [s["processor_wh"] for s in sizes] == [[1024, 512], [352, 1024]]
    # Inspect actual image tensor blocks. Every frame keeps both corners and its original color.
    for path, meta, color in zip(paths, sizes, [(255, 0, 0), (0, 0, 255)], strict=True):
        image, _ = backend.frame(path, 1024, 32)
        x, y = meta["offset_xy"]
        w, h = meta["content_wh"]
        assert image.getpixel((x, y)) == image.getpixel((x + w - 1, y + h - 1)) == color
    patch_count = batch.image_grid_thw[0].prod().item()
    pixels = batch.pixel_values.reshape(-1, 3, 2, 16, 16)
    assert pixels[:patch_count, 0].mean() > pixels[:patch_count, 2].mean()
    assert pixels[patch_count:, 0].mean() < pixels[patch_count:, 2].mean()


class Stub:
    load = {"peak": MEMORY}

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, case):
        if case["task_id"] == self.cfg.get("interrupt"):
            raise RuntimeError("interrupted")
        return {"raw": RAW, "tokens": 17, "peak": MEMORY, "seconds": 1}


def test_resume_and_changed_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "GPU_LOCK", tmp_path / "fake.lock")
    cases = [{"task_id": k, "instruction": k} for k in "abc"]
    folder = tmp_path / "stage"
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.execute(folder, cases, {}, {"interrupt": "b"}, main=False, factory=Stub)
    first = runner.case_path(folder, cases[0]).stat().st_mtime_ns
    changed = [{**cases[0], "instruction": "other"}, *cases[1:]]
    with pytest.raises(ValueError, match="Stale case"):
        runner.execute(folder, changed, {}, {}, main=False, factory=Stub)
    rows = runner.execute(folder, cases, {}, {}, main=False, factory=Stub)
    assert len(rows) == 3 and runner.case_path(folder, cases[0]).stat().st_mtime_ns == first
    assert runner.execute(folder, cases, {}, {}, main=False, factory=None) == rows
    path = runner.case_path(folder, cases[0])
    path.write_text(path.read_text().replace("roof", "bird"))
    with pytest.raises(ValueError, match="Changed output"):
        runner.verified(folder, cases, {})


def test_input_tamper_and_missing_freeze(tmp_path, monkeypatch):
    p = tmp_path / "file"
    p.write_bytes(b"before")
    ref = dict(path=str(p), sha256=protocol.sha256(p))
    p.write_bytes(b"after")
    with pytest.raises(ValueError, match="Frozen input"):
        protocol.checked(ref)
    monkeypatch.setattr(runner, "RELEASE", tmp_path / "no-release")
    with pytest.raises(ValueError, match="committed CPU freeze"):
        runner.check_release({}, [])


def test_report_real_baselines_and_missing_judge_scores(tmp_path, monkeypatch):
    cfg = protocol.read(protocol.CONFIG)
    if not (protocol.ROOT / cfg["images"]["path"]).is_file():
        pytest.skip("Local frozen VN100 evidence unavailable")
    cases, smoke = protocol.datasets(cfg)
    assert len(smoke) == 5
    refs = report.references(cfg, cases)
    common = [r for r in refs if r["graph_consistency"] is not None]
    assert {k: round(report.correlation(common, k, "overall_pass"), 4) for k in report.BASE} == report.BASE
    first = common[0]["task_id"]
    predictions = [
        {"task_id": r["task_id"], "scores": protocol.parse("invalid" if r["task_id"] == first else RAW)}
        for r in reversed(refs)
    ]
    release = tmp_path / "release.json"
    release.write_text("{}")
    monkeypatch.setattr(report, "RELEASE", release)
    (tmp_path / "complete.json").write_text("{}")
    out = report.build(tmp_path, predictions, refs, {**cfg, "bootstrap_replicates": 12}, {})
    summary = protocol.read(out / "summary.json")
    panels = summary["panels"]
    assert [(panels[k]["n"], panels[k]["total"]) for k in ("common91", "all200", "abstain109")] == [
        (90, 91),
        (199, 200),
        (109, 109),
    ]
    assert panels["abstain109"]["sources"] == 71
    stats = panels["common91"]["statistics"]
    assert stats["J"]["estimate"] is None and stats["J_minus_clipscore"]["undefined_replicates"] == 12
    keep = [r for r in common if r["task_id"] != first]
    for k in report.BASE:
        assert stats[k]["estimate"] == report.correlation(keep, k, "overall_pass")
    with (out / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200 and sum(r["status"] == "invalid_format" for r in rows) == 1
    assert "90/91" in (out / "report.md").read_text()


def test_paired_cluster_differences():
    # Each selected source contributes both edits. Equal scores imply exactly zero paired differences.
    rows = [
        dict(
            task_id=str(i),
            image_id=str(i // 2),
            status="ok",
            overall_pass=i % 3 == 0,
            edit_fulfillment=i % 3 == 0,
            human_preservation=True,
            fulfillment=i % 2,
            preservation=4,
            J=i % 2,
            **dict.fromkeys(report.BASE, i % 2),
        )
        for i in range(12)
    ]
    cfg = {"seed": 42, "bootstrap_replicates": 100}
    stats = report.panel(rows, cfg, compare=True)["statistics"]
    for baseline in report.BASE:
        delta = stats["J_minus_" + baseline]
        assert delta["estimate"] == 0 and delta["ci95"] == [0, 0]
        assert delta["valid_replicates"] + delta["undefined_replicates"] == 100
    assert stats["preservation"]["ci95"] is None


def test_cli_smoke_freeze_score_report(tmp_path, monkeypatch):
    import sys

    from tools.level3_mllm_judge_v1 import __main__ as cli

    cfg = protocol.read(protocol.CONFIG)
    if not (protocol.ROOT / cfg["images"]["path"]).is_file():
        pytest.skip("Local frozen VN100 evidence unavailable")
    cases, smoke = protocol.datasets(cfg)
    monkeypatch.setattr(cli, "datasets", lambda _: (cases, smoke))
    monkeypatch.setattr(cli, "identity", lambda _: {"test": True})
    monkeypatch.setattr(cli, "versions", lambda: {})
    run = tmp_path / "judge" / "seed42"
    release = tmp_path / "release.json"
    for mod in (cli, runner):
        monkeypatch.setattr(mod, "RUN", run)
        monkeypatch.setattr(mod, "RELEASE", release)
    monkeypatch.setattr(report, "RELEASE", release)
    monkeypatch.setattr(runner, "GPU_LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(runner, "committed", lambda _: "fake-commit")
    monkeypatch.setattr(cli, "execute", lambda *a, **kw: runner.execute(*a, **kw, factory=Stub))
    events = []

    class Tracker:
        summary = {}

        def log(self, item):
            events.append(item)

        def finish(self, **kwargs):
            pass

    def fake_tracking(folder, header, cfg):
        protocol.atomic(folder / "wandb-id.json", {"id": "local-test"})
        return Tracker()

    monkeypatch.setattr(runner, "tracking", fake_tracking)

    def invoke(name):
        monkeypatch.setattr(sys, "argv", ["judge", name])
        cli.main()

    with pytest.raises(ValueError, match="committed CPU freeze"):
        invoke("score")
    invoke("smoke")
    assert not run.exists() and not release.exists() and not events
    invoke("freeze")
    assert protocol.read(release)["smoke_summary"]["valid"] == 5
    invoke("score")
    assert len(events) == 200 and events[-1]["valid"] == 200
    assert "raw" not in events[-1]
    invoke("score")
    assert len(events) == 200
    assert not (run / "report").exists()
    invoke("report")
    res = protocol.read(run / "report/summary.json")
    assert res["bootstrap"]["replicates"] == 2000
    assert res["panels"]["common91"]["statistics"]["graph_consistency"]["estimate"] == pytest.approx(
        0.1767, abs=0.00005
    )
    assert res["panels"]["all200"]["statistics"]["J"]["undefined_replicates"] == 2000
