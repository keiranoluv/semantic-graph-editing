"""The launcher must retain a paired control for every new fusion family."""

from src.level1_fusion_matrix import CONDITIONS, SEEDS, switches, train_command


def test_fusion_matrix_has_baseline_real_and_shuffled_for_each_family():
    assert len(CONDITIONS) * len(SEEDS) == 21
    assert switches("baseline") == ("off", "none")
    for family, mode in (("pair", "pair"), ("visual", "visual_gate"), ("relation", "relation_prior")):
        assert switches(f"{family}_real") == (mode, "real")
        assert switches(f"{family}_shuffled") == (mode, "shuffled")
        command = train_command(f"{family}_shuffled", 42, "disabled")
        assert "--shuffle-seed" in command
        assert "--kg-relations" in command
