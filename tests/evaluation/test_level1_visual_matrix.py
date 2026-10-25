from src.level1_visual_matrix import CACHE, CONDITIONS, SEEDS, eval_command, train_command


def test_all_conditions_use_same_clip_cache_and_split_budget():
    for condition in CONDITIONS:
        for seed in SEEDS:
            command = train_command(condition, seed, "disabled")
            assert command[command.index("--visual-cache") + 1] == str(CACHE)
            assert command[command.index("--max-train-images") + 1] == "57723"
            assert command[command.index("--max-val-images") + 1] == "5000"
            assert command[command.index("--val-batch-size") + 1] == "512"
            assert command[command.index("--num-workers") + 1] == "0"
            assert command[command.index("--seed") + 1] == str(seed)
            assert ("--no-knowledge" in command) == (condition == "kg_off")
            assert ("--message-edges" in command) == (condition == "edges_off")


def test_independent_validation_uses_batch_512():
    for condition in CONDITIONS:
        for seed in SEEDS:
            command = eval_command(condition, seed, "cuda")
            assert command[command.index("--batch-size") + 1] == "512"
            assert command[command.index("--num-workers") + 1] == "0"
