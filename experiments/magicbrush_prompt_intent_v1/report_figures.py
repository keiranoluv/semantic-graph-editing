"""Regenerate the three Task B figures from the frozen annotations and test results."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.editing.operations import EditGraph
from tools.edit_model_v1.codec import decode_actions
from tools.emnlp2020_graph_modification_v1.dataset import read, sha256
from tools.prepare_magicbrush import ROOT

from .dataset import rows

FIGURES = ROOT / "report_latex/figures"
REPORTS = ROOT / "reports/level2"
INK, GREEN, ORANGE = "#193c65", "#116452", "#9b431d"
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(FIGURES / name, metadata={"CreationDate": None})
    plt.close(fig)


def laptop_graph(row):
    source = EditGraph.from_dict(row["source_graph"])
    nodes = {n.id: n.label for n in source.nodes}
    added_node, added_edge = (op["after"] for op in row["operations"])
    table = added_edge["object"]
    links = [(nodes[e.subject], e.predicate) for e in source.edges if e.object == table and e.predicate == "on"]
    links.append((added_node["label"], added_edge["predicate"]))

    fig, ax = plt.subplots(figsize=(3.48, 1.15))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.01)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.03, 0.94, "Thao tác", weight="bold")
    ax.text(0.23, 0.94, "Đỉnh", weight="bold")
    ax.text(0.48, 0.94, "Quan hệ", weight="bold")
    for (node, predicate), y in zip(links, [0.71, 0.43, 0.15], strict=True):
        added = node == added_node["label"]
        color = GREEN if added else INK
        ax.text(0.03, y, "+ Thêm" if added else "Giữ", color=color, va="center")
        ax.text(0.23, y, node, color=color, va="center", weight="bold")
        ax.annotate(
            "",
            xy=(0.79, 0.43 + 0.19 * (y - 0.43)),
            xytext=(0.42, y),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.25, "shrinkA": 0, "shrinkB": 1},
        )
        ax.text(
            0.53,
            y + (0.43 - y) * 0.30,
            predicate,
            va="center",
            ha="center",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
        )
    ax.text(
        0.80,
        0.43,
        nodes[table],
        ha="left",
        va="center",
        weight="bold",
        color=INK,
        bbox={"boxstyle": "round,pad=.34", "facecolor": "#eaf1f7", "edgecolor": INK},
    )
    save(fig, "magicbrush-laptop-graph.pdf")


def test_errors(test):
    report = read(REPORTS / "magicbrush-prompt-intent-test-v1/summary.json")
    run = next(r for r in report["per_seed"]["prompt"] if r["seed"] == 42)
    path = ROOT / "runs/level2/magicbrush-prompt-intent-v1/heldout-test-v1/prompt-seed-42.json"
    if sha256(path) != run["predictions_sha256"]:
        raise ValueError("Unexpected predictions for prompt/42")
    predictions = {r["id"]: r for r in read(path)}
    fig = plt.figure(figsize=(7.1, 2.58))
    for i, rid in enumerate(["magicbrush:181753:1", "magicbrush:65564:2"]):
        r = test[rid]
        source = EditGraph.from_dict(r["source_graph"])
        pred = decode_actions(predictions[rid]["tokens"], source)
        gold = r["operations"][0]["after"]
        if i == 0:
            before = r["operations"][0]["before"]
            (old,) = set(before["attributes"]) - set(gold["attributes"])
            (new,) = set(gold["attributes"]) - set(before["attributes"])
            assert not pred and r["operations"][0]["kind"] == "replace_node"
            reference = f'{gold["label"]} ({gold["id"]}): {old} → {new}'
            inferred = "∅  (không có thao tác)"
        else:
            add, edge = pred
            assert add.kind == "add_node" and edge.kind == "add_edge" and edge.after.subject == add.after.id
            endpoint = next(n for n in source.nodes if n.id == edge.after.object)
            reference = f'Thêm {gold["label"]}'
            inferred = (
                f"Thêm {add.after.label}; {add.after.label} —{edge.after.predicate}→ {endpoint.label} ({endpoint.id})"
            )
        x = 0.015 + i * 0.50
        fig.text(x, 0.94, f'({chr(97+i)})  {r["instruction"]}', fontsize=9.2, weight="bold", color=INK)
        for j, (role, label) in enumerate([("source_image", "Ảnh nguồn"), ("target_image", "Ảnh đích")]):
            image_path = ROOT / "data/processed/level2/magicbrush-v1/images" / (r["images"][role] + ".png")
            if sha256(image_path) != r["images"][role]:
                raise ValueError(image_path)
            ax = fig.add_axes([x + j * 0.232, 0.30, 0.222, 0.57])
            with Image.open(image_path) as im:
                im.thumbnail((425, 425))
                ax.imshow(im)
            ax.set_title(label, fontsize=7.5, pad=2.4)
            ax.axis("off")
        for y, label, value, color in [(0.20, "Nhãn đích", reference, GREEN), (0.075, "Dự đoán", inferred, ORANGE)]:
            fig.text(x, y, label, color=color, fontsize=7.6, weight="bold")
            fig.text(x + 0.095, y, value, fontsize=7.5)
    save(fig, "magicbrush-test-errors.pdf")


def emnlp_operations():
    report = read(REPORTS / "emnlp2020-task-b-wandb-v4-test-v1/metrics.json")
    metrics = report["models"]["gnn"]["per_type"]
    kinds = ["add_node", "delete_node", "replace_node", "add_edge", "delete_edge", "replace_edge"]
    labels = ["Thêm đỉnh", "Xóa đỉnh", "Thay đỉnh", "Thêm cạnh", "Xóa cạnh", "Thay cạnh"]
    fig, ax = plt.subplots(figsize=(3.48, 1.84))
    fig.subplots_adjust(left=0.36, right=0.85, top=0.98, bottom=0.25)
    for i, kind in enumerate(kinds):
        score = metrics[kind]["f1"]["mean"] * 100
        ax.plot([0, score], [i, i], lw=2, color="#9cb4ca", solid_capstyle="round")
        ax.plot(score, i, "o", color=INK, markersize=4, clip_on=False)
        ax.annotate(
            f"{score:.2f}".replace(".", ","),
            (score, i),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
    ax.set_yticks(
        range(6), [f'{label} ({metrics[k]["support"]["mean"]})' for k, label in zip(kinds, labels, strict=True)]
    )
    ax.set(ylim=(5.4, -0.5), xlim=(0, 100), xticks=[0, 25, 50, 75, 100], xlabel="F1 trung bình (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#dde3e7", linewidth=0.6)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_axisbelow(True)
    save(fig, "emnlp-gnn-operations.pdf")


if __name__ == "__main__":
    laptop_graph(next(r for r in rows("val") if r["id"] == "magicbrush:484847:1"))
    test_errors({r["id"]: r for r in rows("test")})
    emnlp_operations()
