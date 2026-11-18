from types import SimpleNamespace

import pytest
import torch

from ..evaluation import evaluate
from ..m3.common import config
from ..m3.test_m3 import FixtureStore
from .report import diagnostics
from .run import M3Model, M4Model, predict_video, roster


def test_frozen_roster():
    cases = list(roster())
    assert len(cases) == len(set(cases)) == 31
    assert sum(group == "min50" for group, *_ in cases) == 7


@pytest.mark.parametrize("family", ["m3", "m4"])
def test_target_free_timeline_on_synthetic_video(family, tmp_path):
    from ..m4.common import config as m4config

    torch.set_num_threads(2)
    store = FixtureStore()
    vid = next(iter(store.videos))

    def window(*args, supervision):
        assert supervision is False
        return store.window(*args, supervision=supervision)

    inputs = SimpleNamespace(videos=store.videos, window=window)
    cfg = m4config() if family == "m4" else config()
    cfg.update(window_frames=6, validation_stride=3)
    model = M4Model("pvsg_conv_k5_l3", cfg) if family == "m4" else M3Model("st_gat", cfg)
    pred = predict_video(model.eval(), family, inputs, vid, cfg, "cpu")
    report = evaluate([store.videos[vid]], [store.targets[vid]], [pred], expected_video_ids=[vid])
    info = diagnostics(store, [vid], [pred], report, dict(group="baseline", model="static_gat", seed=17), tmp_path)
    assert sorted(i for b in info["fit_frequency_bins"].values() for i in b["predicate_ids"]) == list(range(57))
    assert (tmp_path / "case_timelines.png").is_file()
