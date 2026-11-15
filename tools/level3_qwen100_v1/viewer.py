"""Local source/target review, no automatically inferred human labels."""
import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from tools.level3_image_graph_v1.core import digest, write


def thumb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
        im.thumbnail((640, 480))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render(rows, out, authoring=False):
    cases = []
    for r in rows:
        cases.append(
            {
                "task_id": r["task_id"],
                "prompt": r.get("prompt", r.get("instruction_en")),
                "source": thumb(r["source_path"]),
                "target": thumb(r["target_path"]) if r.get("target_path") else None,
                "status": r.get("status", "not_run"),
                "operation": r.get("operation"),
                "visual_evidence": r.get("visual_evidence"),
            }
        )
    template = {
        "schema": "qwen100-review-v1",
        "results_sha256": digest(rows),
        "reviews": [
            {
                "task_id": r["task_id"],
                "status": "pending",
                "edit_fulfillment": None,
                "preservation": None,
                "reviewer": "",
                "notes": "",
            }
            for r in rows
        ],
    }
    write(out / "review-template.json", template)
    data = (
        json.dumps({"cases": cases, "review": template}, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = """<!doctype html><html lang="vi"><meta charset="utf-8"><title>Qwen · 100 ảnh đích</title>
<style>body{font:16px system-ui;background:#eef3f7;color:#183246;margin:24px}article{background:white;border-radius:12px;padding:16px;margin:16px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}input,button,select{padding:8px;margin:5px}textarea{width:96%;min-height:60px}pre{white-space:pre-wrap}@media(max-width:800px){.pair{display:block}}</style>
<h1>Qwen-Image-2.1 · 100 ảnh nguồn / đích</h1><p>Mỗi ảnh một yêu cầu độc lập. Giữ cả chỉnh sửa sai; chưa tự gán nhãn chất lượng. Preview RGBA ghép nền trắng, PNG gốc được giữ nguyên.</p>
<label>Người duyệt <input id="reviewer"></label><button id="export">Xuất qwen100-reviewed.json</button><button id="previous">Trang trước</button><button id="next">Trang sau</button><span id="page"></span><div id="cases"></div>"""
    html += '<script id="data" type="application/json">' + data + "</script>"
    html += """<script>
const DATA=JSON.parse(document.getElementById('data').textContent);let start=0;const size=10;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){document.getElementById('page').textContent=`${start+1}–${Math.min(start+size,DATA.cases.length)} / ${DATA.cases.length}`;document.getElementById('cases').innerHTML=DATA.cases.slice(start,start+size).map((c,j)=>{const i=start+j,r=DATA.review.reviews[i];return `<article><h2>${i+1}. ${esc(c.task_id)}</h2><p>${esc(c.prompt)}</p><div class="pair"><section>Nguồn<img src="${c.source}"></section><section>Đích · ${esc(c.status)}${c.target?`<img src="${c.target}">`:'Chưa có ảnh đích'}</section></div>${['edit_fulfillment','preservation'].map(k=>`<label>${k==='preservation'?'Bảo toàn':'Đúng yêu cầu'}<select data-i="${i}" data-k="${k}">${[['','Chưa chấm'],['true','Đạt'],['false','Không đạt']].map(([v,t])=>`<option value="${v}" ${(r[k]===null?'':String(r[k]))===v?'selected':''}>${t}</option>`).join('')}</select></label>`).join('')}<textarea data-i="${i}" data-k="notes" placeholder="Nhận xét cụ thể">${esc(r.notes)}</textarea><details><summary>Ý định chỉnh sửa và quan sát ảnh nguồn của assistant</summary><pre>${esc(JSON.stringify({operation:c.operation,visual_evidence:c.visual_evidence},null,2))}</pre></details></article>`}).join('')}
document.getElementById('cases').onchange=e=>{const {i,k}=e.target.dataset;if(i===undefined)return;DATA.review.reviews[i][k]=k==='notes'?e.target.value:e.target.value===''?null:e.target.value==='true'};
document.getElementById('next').onclick=()=>{if(start+size<DATA.cases.length)start+=size;render()};document.getElementById('previous').onclick=()=>{start=Math.max(0,start-size);render()};
document.getElementById('export').onclick=()=>{const name=document.getElementById('reviewer').value.trim();if(!name){alert('Điền tên người duyệt');return}for(const r of DATA.review.reviews){if(r.edit_fulfillment!==null&&r.preservation!==null){if(!r.notes.trim()){alert('Ghi nhận xét cho '+r.task_id);return}r.status='reviewed';r.reviewer=name}}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(DATA.review,null,2)],{type:'application/json'}));a.download='qwen100-reviewed.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></html>"""
    (Path(out) / ("instructions.html" if authoring else "comparison.html")).write_text(html, encoding="utf-8")
