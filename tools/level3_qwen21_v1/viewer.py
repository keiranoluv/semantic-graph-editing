"""Local three-column evidence viewer with initially unscored manual review."""

import base64
import html
import io
import json

from PIL import Image

from tools.level3_image_graph_v1.core import write

from .core import review_template


def picture(path):
    with Image.open(path) as im:
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA")).convert("RGB")
        im.thumbnail((1000, 1000))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=93)
    return '<img src="data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def render(rows, out):
    template = review_template(rows)
    write(out / "review-template.json", template)
    parts = [
        """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Qwen 2.1 · InstructPix2Pix</title>
<style>body{font:16px system-ui;background:#eef2f6;color:#193244;margin:24px}article{background:white;border-radius:12px;padding:18px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{width:95%;min-height:70px}select,input,button{padding:8px;margin:6px}@media(max-width:850px){.grid{display:block}}</style>
<h1>Qwen-Image-2.1 · InstructPix2Pix</h1><p>Cùng ảnh nguồn và câu yêu cầu đã duyệt. Qwen: BF16, seed thực tế 42, 40 bước, CPU offload, output_resolution 1024. IP2P: kết quả cũ, FP16, 30 bước, max side 512, seed thực tế ghi bên dưới. Đây là so sánh chất lượng quan sát, không phải benchmark cùng compute/độ phân giải/seed ảnh.</p>
<p>Chưa có nhãn PASS/FAIL. Tự chấm Edit Fulfillment (đúng yêu cầu) và Preservation (giữ phần không liên quan) cho từng model. Preview ảnh RGBA ghép nền trắng; PNG gốc giữ nguyên alpha. Mở PNG gốc trong thư mục images của các run nếu cần xem chi tiết.</p>
<label>Người duyệt <input id="reviewer"></label><button id="export">Xuất qwen21-reviewed.json</button>"""
    ]
    for i, r in enumerate(rows):
        parts.append(
            "<article><h2>"
            + html.escape(r["task_id"])
            + "</h2><p>"
            + html.escape(r["prompt"])
            + '</p><div class="grid">'
        )
        for tag, title in [("source", "Nguồn"), ("ip2p", "InstructPix2Pix"), ("qwen21", "Qwen-Image-2.1")]:
            parts.append("<section><h3>" + title + "</h3>")
            if r.get(tag + "_path"):
                parts.append(picture(r[tag + "_path"]))
            else:
                parts.append("<p>Không có output: " + html.escape(r.get("error", "chưa chạy")) + "</p>")
            if tag != "source":
                j = 2 * i + (tag == "qwen21")
                for key, title2 in [("edit_fulfillment", "Đúng yêu cầu"), ("preservation", "Bảo toàn")]:
                    parts.append(
                        f'<label>{title2}<select data-i="{j}" data-k="{key}"><option value="">Chưa chấm</option><option value="true">PASS</option><option value="false">FAIL</option></select></label>'
                    )
                parts.append(f'<textarea data-i="{j}" data-k="notes" placeholder="Ghi quan sát cụ thể"></textarea>')
            parts.append("</section>")
        parts.append(
            "</div><details><summary>Cấu hình, thời gian, VRAM và provenance mẫu</summary><pre>"
            + html.escape(json.dumps(r, indent=2, ensure_ascii=False))
            + "</pre></details></article>"
        )
    data = (
        json.dumps(template, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    parts.append('<script id="data" type="application/json">' + data + "</script>")
    parts.append(
        """<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
document.addEventListener('change',e=>{const {i,k}=e.target.dataset;if(i===undefined)return;DATA.reviews[i][k]=k==='notes'?e.target.value:e.target.value===''?null:e.target.value==='true'});
document.getElementById('export').onclick=()=>{const name=document.getElementById('reviewer').value.trim();if(!name){alert('Điền tên người duyệt');return}for(const r of DATA.reviews){if(r.edit_fulfillment!==null&&r.preservation!==null){if(!r.notes.trim()){alert('Ghi quan sát cho '+r.task_id+' / '+r.model);return}r.status='reviewed';r.reviewer=name}}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(DATA,null,2)],{type:'application/json'}));a.download='qwen21-reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
</script></html>"""
    )
    (out / "comparison.html").write_text("".join(parts), encoding="utf-8")
