import numpy as np
import pytest

from src.features.glove import map_vocabulary, read_glove_subset


def test_phrase_policy_and_safe_reproducible_oov():
    vectors = {"traffic": np.array([1.0, 2.0]), "light": np.array([3.0, 4.0])}
    vocab = {"traffic light": 1, "unknown token": 2}
    objects, info = map_vocabulary(vocab, vectors, "object", dimension=2)
    predicates, pred_info = map_vocabulary(vocab, vectors, "predicate", dimension=2)
    np.testing.assert_array_equal(objects[1], vectors["traffic"])
    np.testing.assert_array_equal(predicates[1], [2.0, 3.0])
    assert info["counts"]["random_fallback"] == pred_info["counts"]["random_fallback"] == 1
    assert np.isfinite(predicates).all()
    again, _ = map_vocabulary(dict(reversed(list(vocab.items()))), vectors, "predicate", dimension=2)
    np.testing.assert_array_equal(predicates, again)
    extended, _ = map_vocabulary({**vocab, "extra": 3}, vectors, "predicate", dimension=2)
    np.testing.assert_array_equal(predicates, extended[:3])


def test_exact_and_invalid_vectors(tmp_path):
    path = tmp_path / "glove.txt"
    path.write_text("traffic 1 2\nlight 3 4\nother 5 6\n")
    vectors, rows = read_glove_subset(path, {"traffic"}, dimension=2)
    assert rows == 3 and set(vectors) == {"traffic"}
    a, info = map_vocabulary({"traffic": 1}, vectors, "object", dimension=2)
    assert info["rows"][1]["status"] == "exact" and a.shape == (2, 2)
    path.write_text("traffic nan 2\n")
    with pytest.raises(ValueError):
        read_glove_subset(path, {"traffic"}, dimension=2)
    with pytest.raises(ValueError):
        map_vocabulary({"bad": 2}, {}, "object", dimension=2)
