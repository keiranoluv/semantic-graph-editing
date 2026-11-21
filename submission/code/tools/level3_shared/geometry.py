"""Unchanged matching primitives for Task 3 graph scoring."""


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def label(x):
    return " ".join(x.casefold().split())


def inside(box, region):
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]
