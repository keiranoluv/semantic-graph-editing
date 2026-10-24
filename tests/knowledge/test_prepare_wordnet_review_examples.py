import numpy as np

from scripts.prepare_wordnet_review_examples import sample_examples


def test_samples_only_supplied_train_rows_and_are_deterministic():
    first = np.array([0, 1, 2, 3])
    last = np.array([0, 1, 2, 3])
    labels = np.array([11, 11, 11, 12])
    metadata = [{"image_id": i + 100, "url": f"https://example.com/{i}.jpg"} for i in range(4)]
    a = sample_examples([0, 2, 3], first, last, labels, metadata, seed=7, per_class=2)
    b = sample_examples([0, 2, 3], first, last, labels, metadata, seed=7, per_class=2)
    assert a == b
    assert {item["annotation_row"] for item in a[11]} == {0, 2}
    assert all(item["annotation_row"] != 1 for examples in a.values() for item in examples)
    assert a[12][0]["image_id"] == 103


def test_box_is_first_matching_instance_without_reading_relations():
    first = np.array([0, 2])
    last = np.array([1, 2])
    labels = np.array([11, 11, 12])
    boxes = np.array([[100, 200, 40, 80], [400, 400, 20, 20], [700, 700, 20, 20]])
    metadata = [{"image_id": 10, "width": 1024, "height": 1024}, {"image_id": 11, "width": 1024, "height": 1024}]
    result = sample_examples([0], first, last, labels, metadata, boxes_1024=boxes, per_class=1)
    assert result[11][0]["object_index"] == 0
    assert result[11][0]["box_xyxy"] == [80.0, 160.0, 120.0, 240.0]
