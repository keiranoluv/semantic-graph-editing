"""Small dependency-free SVG diagram for a level 1 PredCls notebook."""

import base64
import html
import math
from pathlib import Path


def scene_svg(graph, relations, object_names, predicate_names, *, width, height, image_path=None, max_nodes=15):
    """Draw boxes and a subset of nodes/relations; annotations remain explicit."""
    shown = min(int(graph.num_nodes), max_nodes)
    image_markup = ""
    if image_path is not None and Path(image_path).is_file():
        encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        image_markup = f'<image href="data:image/jpeg;base64,{encoded}" x="25" y="55" width="560" height="460" preserveAspectRatio="none"/>'
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" viewBox="0 0 1200 560">',
        '<rect width="1200" height="560" fill="white"/>',
        '<text x="25" y="30" font-size="19">Object boxes</text>',
        '<text x="625" y="30" font-size="19">Scene graph (first 15 objects)</text>',
        '<rect x="25" y="55" width="560" height="460" fill="#f5f7fa" stroke="#a0a0a0"/>',
        image_markup,
    ]
    positions = {}
    for node in range(shown):
        box = graph.boxes[node].tolist()
        x = 25 + 560 * box[0] / width
        y = 55 + 460 * box[1] / height
        w = 560 * (box[2] - box[0]) / width
        h = 460 * (box[3] - box[1]) / height
        label = html.escape(object_names[int(graph.object_labels[node])])
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#e07018" stroke-width="2"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{max(y + 16, 65):.1f}" font-size="12" fill="#9a3900">{node}: {label}</text>'
        )
        angle = 2 * math.pi * node / max(shown, 1)
        gx, gy = 910 + 190 * math.cos(angle), 290 + 190 * math.sin(angle)
        positions[node] = (gx, gy)
    displayed = 0
    for source, predicate, target, score in relations:
        if source not in positions or target not in positions or source == target or displayed >= 12:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#7790aa" stroke-width="1.5"/>'
        )
        label = html.escape(predicate_names[predicate])
        if score is not None:
            label += f" ({score:.2f})"
        parts.append(f'<text x="{(x1+x2)/2:.1f}" y="{(y1+y2)/2:.1f}" font-size="11" fill="#204060">{label}</text>')
        displayed += 1
    for node, (x, y) in positions.items():
        label = html.escape(object_names[int(graph.object_labels[node])])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="22" fill="#ecf3fb" stroke="#426f9b"/>')
        parts.append(f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="11">{node}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y+38:.1f}" text-anchor="middle" font-size="11">{label}</text>')
    parts.append(
        f'<text x="25" y="545" font-size="12">Showing {shown}/{graph.num_nodes} nodes and {displayed}/{len(relations)} relations; boxes use GT annotations.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)
