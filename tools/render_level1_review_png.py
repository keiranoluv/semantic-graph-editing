"""Create PNG contact sheets for VG150 PredCls hit/miss review."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

CELL_W, CELL_H = 800, 630
IMAGE_W, IMAGE_H = 760, 480
MARGIN = 20


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def draw_case(sheet: Image.Image, row: dict, index: int, metadata: dict, images_dir: Path) -> None:
    col, line = index % 2, index // 2
    origin_x, origin_y = col * CELL_W, line * CELL_H
    draw = ImageDraw.Draw(sheet)
    image_id = str(row["image_id"])
    if not image_id.isdigit():
        raise ValueError(f"Unsafe image ID: {image_id}")
    with Image.open(images_dir / f"{image_id}.jpg") as source:
        image = ImageOps.contain(source.convert("RGB"), (IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)
    x = origin_x + MARGIN + (IMAGE_W - image.width) // 2
    y = origin_y + 90
    sheet.paste(image, (x, y))
    annotation = metadata[image_id]
    aw, ah = int(annotation["width"]), int(annotation["height"])
    if min(aw, ah) <= 0:
        raise ValueError(f"Invalid image size: {image_id}")
    sx, sy = image.width / aw, image.height / ah
    for key, color, title in (("subject", "#e21b23", "S"), ("object", "#009b50", "O")):
        x1, y1, x2, y2 = row[f"{key}_box_xyxy"]
        box = (x + x1 * sx, y + y1 * sy, x + x2 * sx, y + y2 * sy)
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 3, box[1] + 2), title, fill="white", font=font(16), stroke_width=2, stroke_fill=color)
    subject, obj = row["subject_class"], row["object_class"]
    draw.text(
        (origin_x + MARGIN, origin_y + 12),
        f"Image {image_id}  |  rank {row['predicted_rank']}",
        fill="#17202a",
        font=font(23),
    )
    draw.text(
        (origin_x + MARGIN, origin_y + 47), f"GT: {subject} {row['gt_predicate']} {obj}", fill="#17202a", font=font(19)
    )
    draw.text(
        (origin_x + MARGIN, origin_y + 578),
        f"Pred: {subject} {row['predicted_predicate']} {obj}",
        fill="#17202a",
        font=font(19),
    )
    draw.rectangle((origin_x, origin_y, origin_x + CELL_W - 1, origin_y + CELL_H - 1), outline="#aeb9c4", width=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/vg150/images"))
    parser.add_argument("--image-metadata", type=Path, default=Path("data/raw/vg150/image_data.json"))
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.examples.read_text(encoding="utf-8"))
    metadata = {str(item["image_id"]): item for item in json.loads(args.image_metadata.read_text(encoding="utf-8"))}
    for kind, suffix in (("hit_at_50", "hits"), ("missed_at_50", "misses")):
        output = args.output_prefix.with_name(args.output_prefix.name + f"-{suffix}.png")
        if output.exists():
            raise FileExistsError(output)
        rows = report["examples"][kind]
        sheet = Image.new("RGB", (2 * CELL_W, ((len(rows) + 1) // 2) * CELL_H), "#f4f6f9")
        for index, row in enumerate(rows):
            draw_case(sheet, row, index, metadata, args.images_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, optimize=True)
        print(f"[OK] {len(rows)} cases: {output}")


if __name__ == "__main__":
    main()
