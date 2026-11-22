"""Bounding-box overlays for the saved VN100 examples."""

import html


def graph_svg(graph, region=None):
    w, h = graph["image"]["width"], graph["image"]["height"]
    shapes = []
    for key, node in graph["grounding"]["nodes"].items():
        x, y, x2, y2 = node["bbox_xyxy"]
        title = html.escape(f'{key}: {node["label"]}, confidence={node["confidence"]}')
        shapes.append(
            f'<g><title>{title}</title><rect x="{x}" y="{y}" width="{x2-x}" height="{y2-y}"/>'
            f'<text x="{x}" y="{max(y, h/40)}" font-size="{max(w,h)/55}">{html.escape(key+": "+node["label"])}</text></g>'
        )
    if region:
        x, y, x2, y2 = region
        shapes.append(
            f'<rect class="roi" x="{x*w}" y="{y*h}" width="{(x2-x)*w}" height="{(y2-y)*h}"><title>Vùng edit định trước</title></rect>'
        )
    return f'<svg class="boxes" viewBox="0 0 {w} {h}" aria-label="Bounding boxes REACT++">' + "".join(shapes) + "</svg>"
