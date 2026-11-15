"""CPU contracts. Synthetic predictions below are fixtures, never measured quality."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools.level3_image_graph_v1.core import digest, read, sha256, validate_record, write
from tools.level3_sgg_compare_v1 import adapter, backend, comparison, runner

masks = pytest.importorskip(
    "pycocotools.mask", reason="Use requirements-level3-sgg-robin-v1.txt in an isolated environment"
)

TEXT = "Objects:\nregion2: a red bicycle\nregion1: a man wearing a red shirt\nRelations:\nregion1: region2 riding\nregion2: region1 under"


def region(identifier=1, width=100, height=60):
    binary = np.zeros((height, width), dtype=np.uint8)
    binary[2 : height - 2, 3 : width - 3] = 1
    rle = masks.encode(np.asfortranarray(binary))
    rle["counts"] = rle["counts"].decode("ascii")
    return {
        "region_id": identifier,
        "bbox_xyxy": [3, 2, width - 3, height - 2],
        "segmentation": rle,
        "predicted_iou": 0.94,
        "stability_score": 0.98,
        "origin": "synthetic_test_fixture",
    }


@pytest.fixture
def row():
    return {
        "image_id": "sidewalk-fixture",
        "width": 100,
        "height": 60,
        "sha256": "a" * 64,
        "image_path": "images/sidewalk-fixture.png",
        "dataset": "sidewalk",
        "license": {"note": "metadata must survive"},
    }


def record(row):
    result = adapter.robin_record(row, [region(), region(2)], TEXT, {"fixture": True}, {"hit_token_limit": False})
    result["status"] = "ok"
    return result


def test_native_numbering_direction_attributes_and_confidence(row):
    result = record(row)
    graph = validate_record(result)
    assert graph.nodes[0].label == "a man wearing a red shirt"
    assert graph.nodes[0].attributes == ()  # no borrowed shirt color
    assert graph.nodes[1].attributes == ("red",)
    assert [(e.subject, e.predicate, e.object) for e in graph.edges] == [
        ("n001", "riding", "n002"),
        ("n002", "under", "n001"),
    ]
    assert result["image"] == row
    assert result["native_output"]["text"] == TEXT
    assert result["grounding"]["nodes"]["n001"]["confidence"] is None
    assert result["grounding"]["nodes"]["n001"]["region"]["predicted_iou"] == 0.94
    assert result["human_reviewed"] is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Objects:\nregion1: man",
        TEXT.replace("region2: a red bicycle\n", ""),
        TEXT.replace("region2 riding", "region3 riding"),
        TEXT.replace("region1: a man", "region2: a man"),
        TEXT + "\nhallucinated trailing commentary",
        TEXT.replace("region2 riding", "region2 riding; region1 holding"),
        TEXT.replace("a red bicycle", "a red bicycle\nunparsed description"),
    ],
)
def test_malformed_native_is_failure_not_empty_graph(text):
    with pytest.raises(adapter.ParseError):
        adapter.parse_robin(text, 2)


def test_duplicate_edges_retained_as_evidence_and_no_invented_reverse():
    text = "Objects:\nregion1: person\nregion2: bike\nRelations:\nregion1: region2 riding, region2 riding"
    objects, edges, duplicates = adapter.parse_robin(text, 2)
    assert objects == {1: "person", 2: "bike"}
    assert edges == [(1, "riding", 2)] and len(duplicates) == 1
    assert adapter.parse_robin("Objects:\nregion1: tree\nRelations: none", 1)[1] == []


@pytest.mark.parametrize(
    "text,attrs",
    [
        ("a small red car", ["red", "small"]),
        ("a man with a black hat", []),
        ("a glass", []),
        ("a wooden table", ["wooden"]),
        ("red, green signs", ["red"]),
    ],
)
def test_prefix_projection_ownership(text, attrs):
    assert adapter.description_attributes(text) == attrs


def test_truncated_native_never_passes_as_graph(row):
    with pytest.raises(adapter.ParseError, match="token_limit"):
        adapter.robin_record(row, [region(), region(2)], TEXT, {}, {"hit_token_limit": True})


@pytest.mark.parametrize("width,height,offset", [(100, 60, [0, 20]), (60, 100, [20, 0]), (60, 60, [0, 0])])
def test_rgb_mask_and_box_share_padding(width, height, offset):
    image = Image.new("RGB", (width, height), (255, 0, 0))
    r = region(width=width, height=height)
    canvas, binary, boxes, evidence = backend.square_inputs(image, [r])
    side = max(width, height)
    dx, dy = offset
    assert canvas.size == (side, side) and binary.shape == (1, side, side)
    assert canvas.getpixel((dx, dy)) == (255, 0, 0)
    expected = masks.decode({**r["segmentation"], "counts": r["segmentation"]["counts"].encode()})
    assert np.array_equal(binary[0, dy : dy + height, dx : dx + width], expected)
    assert binary.sum() == expected.sum()
    assert boxes == [[int(1000 * v / side) for v in (3 + dx, 2 + dy, width - 3 + dx, height - 2 + dy)]]
    assert evidence["offset_xy"] == offset


def test_region_budget_retains_original_indices_masks_and_exclusions():
    native = []
    for area in (100, 500, 600, 5900):
        native.append(
            {
                "bbox": [3, 2, 94, 56],
                "area": area,
                "stability_score": 0.98,
                "predicted_iou": 0.93,
                "segmentation": region()["segmentation"],
            }
        )
    selected, excluded = backend.select_regions(
        native, (100, 60), {"min_area_fraction": 0.02, "max_area_fraction": 0.85, "max_regions": 1}
    )
    assert selected[0]["sam_index"] == 2 and selected[0]["region_id"] == 1
    assert selected[0]["native_region_sha256"] == digest(native[2])
    assert {e["sam_index"]: e["reason"] for e in excluded} == {0: "area_filter", 1: "region_budget", 3: "area_filter"}


def five_rows(row):
    return [{**row, "image_id": f"sidewalk-{i}"} for i in range(5)]


def test_comparison_unrun_is_null_not_zero_and_identity_is_strict(row):
    rows = five_rows(row)
    records = [record(r) for r in rows]
    result, review = comparison.make_comparison(rows, records, [])
    assert result["cases"][0]["models"]["robin-sam"]["statistics"]["edges"] is None
    assert len(review["reviews"]) == 10
    with pytest.raises(ValueError, match="same_five"):
        comparison.make_comparison(rows, records, records[:4])
    changed = copy.deepcopy(records)
    changed[0]["image"]["license"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        comparison.make_comparison(rows, records, changed)


def test_quality_reviews_are_evidence_bound_and_cannot_score_unrun(row):
    rows = five_rows(row)
    result, review = comparison.make_comparison(rows, [record(r) for r in rows], [])
    summary = comparison.validate_quality(result, review)
    assert not summary["complete_paired_review"]
    review["reviews"][1].update(
        status="reviewed", reviewer="fixture", notes="fixture", scores={a: 2 for a in comparison.AXES}
    )
    with pytest.raises(ValueError, match="missing_prediction"):
        comparison.validate_quality(result, review)
    review["comparison_sha256"] = "bad"
    with pytest.raises(ValueError, match="stale"):
        comparison.validate_quality(result, review)


def test_quality_complete_pairs_reject_boolean_and_stale_record(row):
    rows = five_rows(row)
    records = [record(r) for r in rows]
    result, review = comparison.make_comparison(rows, records, records)
    for r in review["reviews"]:
        r.update(status="reviewed", reviewer="fixture", notes="fixture", scores={a: 2 for a in comparison.AXES})
    assert comparison.validate_quality(result, review)["complete_paired_review"]
    review["reviews"][0]["scores"]["overall"] = True
    with pytest.raises(ValueError, match="0_1_2"):
        comparison.validate_quality(result, review)
    review["reviews"][0]["record_sha256"] = "bad"
    with pytest.raises(ValueError, match="evidence"):
        comparison.validate_quality(result, review)


@pytest.fixture
def cohort(tmp_path, row):
    locked, baseline = tmp_path / "locked", tmp_path / "baseline"
    rows, tasks = [], []
    for group in ("sidewalk", "travel"):
        for i in range(15):
            r = {**row, "image_id": f"{group}-{i:02d}", "dataset": group, "image_path": f"images/{group}-{i:02d}.png"}
            path = locked / r["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (100, 60), (i, 50, 80)).save(path)
            r["sha256"] = sha256(path)
            rows.append(r)
            tasks.append(
                {
                    "image_id": r["image_id"],
                    "source_sha256": r["sha256"],
                    "image_path": r["image_path"],
                    "source_graph": None,
                }
            )
    write(locked / "manifest.json", {"image_count": 30, "images": rows})
    (locked / "source_graph_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks))
    write(
        baseline / "candidates.json",
        {"schema": "level3-image-graph-v1", "run": {"mode": "infer"}, "records": [record(r) for r in rows]},
    )
    write(
        baseline / "receipt.json",
        {"complete": True, "files_sha256": {"candidates.json": sha256(baseline / "candidates.json")}},
    )
    cfg = read(runner.CONFIG)
    chosen = rows[:3] + rows[15:17]
    cfg.update(
        images=[{"candidate_id": r["image_id"], "sha256": r["sha256"]} for r in chosen],
        locked_manifest_sha256=sha256(locked / "manifest.json"),
        locked_tasks_sha256=sha256(locked / "source_graph_tasks.jsonl"),
        baseline_candidates_sha256=sha256(baseline / "candidates.json"),
        baseline_receipt_sha256=sha256(baseline / "receipt.json"),
    )
    config = tmp_path / "config.json"
    write(config, cfg)
    return SimpleNamespace(
        locked=locked,
        baseline=baseline,
        config=config,
        cache=tmp_path / "cache",
        output=tmp_path / "output",
        device="cpu",
    )


def test_cohort_identity_and_output_protection(cohort):
    _, rows, base, _ = runner.inputs(cohort)
    assert len(rows) == len(base) == 5
    assert [r["image_id"] for r in rows] == ["sidewalk-00", "sidewalk-01", "sidewalk-02", "travel-00", "travel-01"]
    for root in (cohort.locked, cohort.baseline, cohort.cache):
        with pytest.raises(ValueError, match="separate"):
            runner.protected_output(root / "new", cohort)
    (cohort.baseline / "candidates.json").write_text("{}")
    with pytest.raises(ValueError, match="baseline_revision"):
        runner.inputs(cohort)


@pytest.mark.parametrize("invalid_one", [False, True])
def test_cpu_fixture_pipeline_preserves_failures_and_unloads_sam(cohort, monkeypatch, invalid_one):
    events = []

    class Sam:
        def __del__(self):
            events.append("sam_unloaded")

    class Robin:
        loading_evidence = {"fixture": True}

        def __init__(self, *args):
            assert events == ["sam"] * 5 + ["sam_unloaded"]
            self.index = 0

        def generate(self, *args):
            self.index += 1
            events.append("robin")
            return ("malformed" if invalid_one and self.index == 2 else TEXT), {"hit_token_limit": False}

    def propose(*args):
        events.append("sam")
        return {"regions": [region(), region(2)], "native_regions": [], "excluded": []}

    monkeypatch.setattr(runner, "verify_assets", lambda *a: {"fixture": True})
    monkeypatch.setattr(runner, "runtime_versions", lambda: {"fixture": True})
    monkeypatch.setattr(runner, "load_sam", lambda *a: Sam())
    monkeypatch.setattr(runner, "propose_regions", propose)
    monkeypatch.setattr(runner, "RobinBackend", Robin)
    assert runner.infer(cohort) == (2 if invalid_one else 0)
    runner.verify_receipt(cohort.output)
    result = read(cohort.output / "robin.json")["records"]
    assert len(result) == 5 and len(list((cohort.output / "native").glob("*.json"))) == 5
    assert result[1]["status"] == ("parse_error" if invalid_one else "ok")
    if invalid_one:
        assert result[1]["source_graph"] is None
    else:
        assert all(validate_record(r) for r in result)
    assert read(cohort.output / "receipt.json")["human_quality_measured"] is False
    with pytest.raises(FileExistsError):
        runner.infer(cohort)


def test_cli_baseline_and_offline_html(cohort):
    args = [
        "--config",
        str(cohort.config),
        "--locked",
        str(cohort.locked),
        "--baseline",
        str(cohort.baseline),
        "--cache",
        str(cohort.cache),
        "--output",
        str(cohort.output),
    ]
    assert runner.main(["baseline", *args]) == 0
    runner.verify_receipt(cohort.output)
    html = (cohort.output / "comparison.html").read_text()
    assert "data:image/jpeg;base64," in html and '<script src="http' not in html
    assert "not_run" in html and "n001" in html
    result = subprocess.run(
        [sys.executable, "-m", "tools.level3_sgg_compare_v1", "smoke", "--device", "cuda"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_source_verification_rejects_changed_or_extra_code(tmp_path):
    from tools.level3_sgg_compare_v1.assets import verify_source

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "model.py"
    path.write_text("# fixture\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "model.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=CPU test",
            "-c",
            "user.email=cpu@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    assert verify_source(tmp_path, {"commit": commit})["commit"] == commit
    (tmp_path / "extra.py").write_text("# unexpected")
    with pytest.raises(ValueError, match="unexpected"):
        verify_source(tmp_path, {"commit": commit})
    (tmp_path / "extra.py").unlink()
    path.write_text("# changed")
    with pytest.raises(ValueError, match="changed"):
        verify_source(tmp_path, {"commit": commit})


def test_gamma_restore_is_exact_and_rejects_unrelated_missing_weights(tmp_path):
    import torch
    from safetensors.torch import save_file

    key = "model.vision_tower.vision_tower.visual.trunk.stages.0.blocks.0.gamma"
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    original = torch.tensor([0.25, 0.5, 0.75], dtype=torch.bfloat16)
    model = SimpleNamespace(named_parameters=lambda: [(key, parameter)])
    save_file({key: original}, str(tmp_path / "part.safetensors"))
    write(tmp_path / "model.safetensors.index.json", {"weight_map": {key: "part.safetensors"}})
    info = {
        "missing_keys": [key],
        "unexpected_keys": [key.removesuffix("gamma") + "weight"],
        "mismatched_keys": [],
        "error_msgs": [],
    }
    evidence = backend.restore_convnext_gamma(model, tmp_path, info)
    assert torch.equal(parameter, original)
    assert evidence[0]["verified_exact_after_dtype_conversion"]
    info["missing_keys"].append("lm_head.weight")
    with pytest.raises(ValueError, match="unrecognized"):
        backend.restore_convnext_gamma(model, tmp_path, info)


def test_html_escapes_model_text(cohort):
    _, rows, records, _ = runner.inputs(cohort)
    records[0]["source_graph"]["nodes"][0]["label"] = '<script>alert("unsafe")</script>'
    result, _ = comparison.make_comparison(rows, records, [])
    destination = cohort.output.with_suffix(".html")
    comparison.render_comparison(result, cohort.locked, destination)
    text = destination.read_text()
    assert "<script>alert(" not in text and "&lt;script&gt;" in text


@pytest.mark.parametrize(
    "capability,arches",
    [
        ((12, 0), ["sm_75", "sm_80", "sm_90"]),
        ((12, 0), ["compute_120"]),
        ((9, 0), ["sm_90a"]),
    ],
)
def test_cuda_arch_rejected_without_kernels(capability, arches):
    from tools.level3_sgg_compare_v1.cuda_check import validate_cuda_arch

    with pytest.raises(RuntimeError, match="cuda_arch_not_supported"):
        validate_cuda_arch(capability, arches)


@pytest.mark.parametrize("capability,arches", [((12, 0), ["sm_90", "sm_120"]), ((8, 6), ["sm_80"])])
def test_cuda_arch_native_binary_support(capability, arches):
    from tools.level3_sgg_compare_v1.cuda_check import validate_cuda_arch

    validate_cuda_arch(capability, arches)


def test_runtime_profiles_require_matching_cuda128_pair():
    from tools.level3_sgg_compare_v1.assets import torch_runtime_profile

    assert torch_runtime_profile("2.7.1+cu128", "0.22.1+cu128") == "robin-cu128-v2"
    assert torch_runtime_profile("2.5.1+cu124", "0.20.1+cu124") == "robin-legacy-v1"
    for pair in (("2.7.1+cu126", "0.22.1+cu126"), ("2.7.1+cu128", "0.20.1+cu124")):
        with pytest.raises(ValueError, match="unsupported_torch_torchvision_pair"):
            torch_runtime_profile(*pair)


def test_unsupported_gpu_fails_before_output_or_checkpoint_load(cohort, monkeypatch):
    cohort.device = "cuda"

    def rejected():
        raise RuntimeError("cuda_arch_not_supported")

    def unexpected(*args):
        pytest.fail("checkpoint work started on an unsupported GPU")

    monkeypatch.setattr(runner, "cuda_device_evidence", rejected)
    monkeypatch.setattr(runner, "verify_assets", unexpected)
    monkeypatch.setattr(runner, "load_sam", unexpected)
    with pytest.raises(RuntimeError, match="cuda_arch_not_supported"):
        runner.infer(cohort)
    assert not cohort.output.exists()
