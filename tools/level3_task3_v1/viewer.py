"""Standalone authoring and blinded outcome review; JSON export stays local."""

import base64
import html
import io
import json

from PIL import Image

from tools.level3_image_graph_v1.core import DEFAULT_LOCKED, digest, read, write
from tools.level3_sgg_pretrained_v2.comparison import graph_svg


def thumbnail(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1000, 750))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=87)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def document(title, data, body, script):
    text = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (
        '<!doctype html><html lang="vi"><meta charset="utf-8"><title>'
        + title
        + "</title><style>body{font:16px system-ui;background:#eef3f7;color:#1c3545;margin:24px}article{background:white;padding:20px;margin:20px 0;border-radius:12px}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}textarea{display:block;width:98%;min-height:70px;margin:8px 0}input,select,button{padding:8px;margin:4px}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{cursor:pointer}details{margin:12px 0}@media(max-width:800px){.pair{display:block}}</style><h1>"
        + title
        + "</h1>"
        + body
        + '<script id="data" type="application/json">'
        + text
        + "</script><script>"
        + script
        + "</script></html>"
    )


COMMON_JS = """
const DATA=JSON.parse(document.getElementById('data').textContent);
function download(obj,name){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(obj,null,2)],{type:'application/json'}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
"""


def authoring(pack, records, output, context):
    packet = {
        "schema": "level3-task3-authoring-v1",
        "context": context,
        "draft_sha256": digest(pack),
        "pack": pack,
        "images": {k: thumbnail(DEFAULT_LOCKED / r["image"]["image_path"]) for k, r in records.items()},
        "graphs": {
            k: {"source_graph": r["source_graph"], "grounding": r["grounding"], "image": r["image"]}
            for k, r in records.items()
        },
    }
    write(output / "draft.json", pack)
    write(output / "authoring-context.json", {"context": context, "draft_sha256": digest(pack)})
    body = '<p>5 ảnh · 10 câu nháp. Duyệt câu yêu cầu, mô tả trạng thái đích và object/region mục tiêu. Mỗi yêu cầu độc lập từ ảnh nguồn. Bản nháp chưa phải nhãn tay.</p><p>Node ID và bbox trong chi tiết giúp kiểm tra đúng đối tượng; vùng thêm dùng tọa độ 0–1. Không đổi câu yêu cầu sang nội dung khác mà giữ operation cũ.</p><label>Người duyệt <input id="reviewer"></label><button id="export">Xuất tasks-reviewed.json</button><input type="file" id="import" accept="application/json"><div id="cases"></div><p id="notice"></p>'
    script = (
        COMMON_JS
        + """
const tasks=DATA.pack.tasks;
function picture(t){const g=DATA.graphs[t.image_id],im=g.image;let b=t.operation.source_node?g.grounding.nodes[t.operation.source_node]?.bbox_xyxy:null;if(t.operation.kind==='add')b=t.operation.region.map((x,i)=>x*(i%2?im.height:im.width));return `<svg style="width:100%" viewBox="0 0 ${im.width} ${im.height}"><image href="${DATA.images[t.image_id]}" width="${im.width}" height="${im.height}"/>${b?`<rect x="${b[0]}" y="${b[1]}" width="${b[2]-b[0]}" height="${b[3]-b[1]}" fill="none" stroke="#ffe000" stroke-width="6"/>`:''}</svg>`}

function render(){document.getElementById('cases').innerHTML=tasks.map((t,i)=>`<article><h2>${i+1}. ${esc(t.task_id)}</h2>${picture(t)}<label>Câu chỉnh sửa (English)<textarea data-i="${i}" data-k="instruction_en">${esc(t.instruction_en)}</textarea></label><label>Mô tả ảnh đích cho CLIPScore — viết trước khi xem output<textarea data-i="${i}" data-k="target_caption_en">${esc(t.target_caption_en)}</textarea></label><details><summary>Đối tượng và quan hệ nguồn (dự đoán REACT++, chưa phải GT)</summary><pre>${esc(JSON.stringify(DATA.graphs[t.image_id],null,2))}</pre></details><label>Thao tác kỳ vọng — kiểm tra source_node hoặc region<textarea data-i="${i}" data-k="operation">${esc(JSON.stringify(t.operation))}</textarea></label><label>Ghi chú<textarea data-i="${i}" data-k="notes">${esc(t.notes)}</textarea></label><label><input type="checkbox" data-i="${i}" data-k="approved" ${t.status==='approved'?'checked':''}> Tôi đã kiểm tra yêu cầu và thao tác khớp với ảnh</label></article>`).join('')}
render();
document.getElementById('cases').addEventListener('change',e=>{const i=e.target.dataset.i,k=e.target.dataset.k;if(i===undefined)return;try{if(k==='approved')tasks[i].status=e.target.checked?'approved':'draft';else if(k==='operation')tasks[i][k]=JSON.parse(e.target.value);else tasks[i][k]=e.target.value;e.target.style.outline='';delete e.target.dataset.invalid;if(k==='operation')render()}catch(err){e.target.dataset.invalid='true';e.target.style.outline='2px solid red';alert('Operation JSON chưa hợp lệ')}});
document.getElementById('export').onclick=()=>{if(document.querySelector('[data-invalid]')){alert('Sửa operation JSON trước khi xuất');return}const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer||tasks.some(t=>t.status!=='approved')){alert('Điền tên và duyệt tất cả yêu cầu trước khi xuất.');return}tasks.forEach(t=>t.reviewer=reviewer);download({schema:DATA.schema,context:DATA.context,draft_sha256:DATA.draft_sha256,pack:{schema:DATA.pack.schema,tasks}},'tasks-reviewed.json')};
document.getElementById('import').onchange=async e=>{try{const x=JSON.parse(await e.target.files[0].text());if(JSON.stringify(x.context)!==JSON.stringify(DATA.context)||x.draft_sha256!==DATA.draft_sha256)throw Error('File khác phiên bản');tasks.splice(0,tasks.length,...x.pack.tasks);document.getElementById('reviewer').value=tasks[0].reviewer;render()}catch(err){alert(err.message)}};
"""
    )
    (output / "authoring.html").write_text(
        document("Task 3 · Duyệt yêu cầu chỉnh sửa", packet, body, script), encoding="utf-8"
    )


def outcome_review(results, output, smoke=False):
    labels = []
    cases = []
    for r in results:
        labels.append(
            {
                "case_id": r["case_id"],
                "result_sha256": digest(r),
                "status": "pending",
                "reviewer": "",
                "notes": "",
                "edit_pass": None,
                "preservation_pass": None,
                "overall_pass": None,
            }
        )
        cases.append(
            {
                "case_id": r["case_id"],
                "instruction_en": r["instruction_en"],
                "status": r["status"],
                "source": thumbnail(r["source_path"]),
                "target": thumbnail(r["target_path"]) if r.get("target_path") else None,
            }
        )
    review = {"schema": "level3-task3-outcome-review-v1", "results_sha256": digest(results), "reviews": labels}
    write(output / "outcome-review-template.json", review)
    body = (
        ("<p>Smoke kỹ thuật; không dùng làm kết quả benchmark.</p>" if smoke else "")
        + '<p>Chấm từ ảnh và câu yêu cầu. Điểm tự động được ẩn để hạn chế ảnh hưởng tới nhãn tay. Đạt chung chỉ khi cả chỉnh sửa và bảo toàn đều đạt. Chưa chấm giữ trống.</p><label>Người duyệt <input id="reviewer"></label><button id="export">Xuất outcome-reviewed.json</button><div id="cases"></div>'
    )
    script = (
        COMMON_JS
        + """
document.getElementById('cases').innerHTML=DATA.cases.map((c,i)=>`<article><h2>${esc(c.case_id)}</h2><p>${esc(c.instruction_en)} · ${esc(c.status)}</p><div class="pair"><section>Ảnh nguồn<img src="${c.source}"></section><section>Ảnh chỉnh sửa${c.target?`<img src="${c.target}">`:'Không có output'}</section></div>${['edit_pass','preservation_pass'].map(k=>`<label>${k==='edit_pass'?'Thực hiện đúng yêu cầu':'Bảo toàn phần không liên quan'}<select data-i="${i}" data-k="${k}"><option value="">Chưa chấm</option><option value="true">Đạt</option><option value="false">Không đạt</option></select></label>`).join('')}<textarea data-i="${i}" data-k="notes" placeholder="Ghi rõ thay đổi đúng/sai hoặc phần bị phá hỏng"></textarea></article>`).join('');
document.getElementById('cases').onchange=e=>{const i=e.target.dataset.i,k=e.target.dataset.k;if(i===undefined)return;DATA.review.reviews[i][k]=k==='notes'?e.target.value:e.target.value===''?null:e.target.value==='true'};
document.getElementById('export').onclick=()=>{const name=document.getElementById('reviewer').value.trim();if(!name){alert('Điền tên người duyệt');return}for(const r of DATA.review.reviews){if(r.edit_pass!==null&&r.preservation_pass!==null){if(!r.notes.trim()){alert('Cần ghi chú cho '+r.case_id);return}r.status='reviewed';r.reviewer=name;r.overall_pass=r.edit_pass&&r.preservation_pass}}download(DATA.review,'outcome-reviewed.json')};
"""
    )
    (output / "review.html").write_text(
        document("Task 3 · Đánh giá ảnh trước/sau", {"cases": cases, "review": review}, body, script), encoding="utf-8"
    )
    # Separate evidence page: users can inspect scores after recording blinded labels.
    parts = [
        '<!doctype html><html lang="vi"><meta charset="utf-8"><style>body{font:16px system-ui;margin:24px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img,svg{width:100%}pre{white-space:pre-wrap}@media(max-width:800px){.pair{display:block}}</style><h1>Task 3 · Điểm và bằng chứng graph</h1><p>Đây là điểm heuristic từ graph dự đoán, không phải nhãn tay.</p>'
    ]
    for r in results:
        parts.append("<h2>" + html.escape(r["case_id"]) + '</h2><div class="pair">')
        for tag in ("source", "target"):
            parts.append("<section><h3>" + tag + "</h3>")
            if r.get(tag + "_path"):
                parts.append('<img src="' + thumbnail(r[tag + "_path"]) + '">')
            if r.get(tag + "_graph_path"):
                record = read(r[tag + "_graph_path"])
                graph = record["source_graph"]
                parts.append(graph_svg(graph, r["case_id"] + "-" + tag))
                parts.append(
                    "<details><summary>Node, bbox và quan hệ có hướng</summary><pre>"
                    + html.escape(
                        json.dumps({"graph": graph, "grounding": record["grounding"]}, ensure_ascii=False, indent=2)
                    )
                    + "</pre></details>"
                )
            parts.append("</section>")
        parts.append("</div>")
        parts.append(
            "<h2>"
            + html.escape(r["case_id"])
            + '</h2><pre style="white-space:pre-wrap">'
            + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
            + "</pre>"
        )
    (output / "scores.html").write_text("".join(parts) + "</html>", encoding="utf-8")
