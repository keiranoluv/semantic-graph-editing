"use strict";
const $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const ui = {boot:null, data:null, state:null, phase:0, selected:{source:null,target:null}, dirty:false, version:0, timer:null, validationTimer:null, chain:Promise.resolve(), busy:false, conflict:false, undo:[], redo:[], lastGroup:null};
const statusNames = {draft:"Chưa xong",source_locked:"Đã có nguồn",completed:"Đã hoàn thành",deferred:"Tạm bỏ qua"};
const roleName = role => role === "source" ? "nguồn" : "đích";
const number = id => (id.startsWith("newn") ? "+" : "") + Number(id.replace(/\D/g,""));
const editable = role => !ui.busy && !ui.conflict && (role === "source" ? !ui.state.source_locked : ui.state.source_locked);
const graph = role => ui.state[role + "_graph"];
const localKey = () => "magicbrush-manual:" + ui.boot.workspace_sha256 + ":" + ui.data.id;
const profileKey = "magicbrush-manual:annotator";
const attrs = node => node.attributes.length ? " [" + node.attributes.join(", ") + "]" : "";
function el(tag, text, className) { const item=document.createElement(tag); if(text!==undefined) item.textContent=text; if(className) item.className=className; return item; }
function notice(text, error=false) { $("notice").replaceChildren(); $("notice").textContent=text; $("notice").hidden=!text; $("notice").classList.toggle("error",error); }
function saved(text, error=false) { $("save-state").textContent=text; $("save-state").className="save-state "+(error?"error":"saved"); }
async function api(path, body) {
  const response=await fetch(path,body===undefined?{cache:"no-store"}:{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const result=await response.json();
  if(!response.ok) { const error=new Error(result.error||"Không kết nối được máy chủ."); error.status=response.status; throw error; }
  return result;
}
function download(name,value) { const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:"application/json"})); const a=el("a"); a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000); }
function snapshot() { const result={}; for(const key of ["source_graph","target_graph","source_points","target_points","notes","reason","instruction_matches","unchanged_checked"]) result[key]=clone(ui.state[key]); return result; }
function persistLocal() {
  try { localStorage.setItem(localKey(),JSON.stringify({base_revision:ui.state.revision,source_locked:ui.state.source_locked,annotator:ui.state.annotator,...snapshot()})); }
  catch { notice("Bộ nhớ trình duyệt đã đầy. Hãy giữ kết nối máy chủ hoặc tải bản nháp JSON.",true); }
}
function changed(fn, render=true, group=null) {
  if(ui.busy || ui.conflict) return;
  if(!group || ui.lastGroup?.key!==group || Date.now()-ui.lastGroup.time>1200) { ui.undo.push(snapshot()); if(ui.undo.length>80) ui.undo.shift(); }
  ui.lastGroup={key:group,time:Date.now()}; ui.redo=[]; fn(); ui.version++;ui.dirty=true;
  if(ui.state.status==="completed") {ui.state.status="source_locked";ui.boot.cases.find(c=>c.id===ui.data.id).status="source_locked";renderPhase();renderProgress();}
  persistLocal();saved("Có thay đổi · đang lưu…");
  clearTimeout(ui.timer);ui.timer=setTimeout(()=>save().catch(showError),700);
  if(render) renderGraphs(); else { renderPins("source");if(ui.state.source_locked) renderPins("target"); }
  renderButtons();scheduleValidation();
}
function showError(error) {
  saved("Chưa lưu được",true);notice(error.message,true);
  if(error.status===409) {ui.conflict=true;renderGraphs();renderButtons();const button=el("button","Tải lại bản trên máy chủ","small-button");button.onclick=()=>loadCase(ui.data.id,true).catch(showError);$("notice").append(document.createTextNode(" "),button);}
}
function payload() { return {base_revision:ui.state.revision,annotator:$("annotator").value,...snapshot()}; }
function save(action="draft") {
  clearTimeout(ui.timer);
  ui.chain=ui.chain.catch(()=>{}).then(async()=>{
    if(ui.conflict) throw new Error("Hãy xử lý bản nháp đang xung đột trước khi lưu.");
    if(action==="draft"&&!ui.dirty) return ui.data;
    const version=ui.version, id=ui.data.id;
    const result=await api("/api/save",{id,action,payload:payload()});
    if(ui.data.id!==id) return result;
    ui.data=result;
    if(action!=="draft"||ui.version===version) {
      ui.state=clone(result.state);ui.dirty=false;localStorage.removeItem(localKey());
    } else {
      for(const key of ["revision","event_sha256","status"]) ui.state[key]=result.state[key];
      persistLocal();
    }
    ui.boot.cases.find(c=>c.id===id).status=result.state.status;
    renderQueue();renderProgress();
    saved(ui.dirty?"Có thay đổi · đang lưu…":"Đã lưu vào máy · lượt "+result.state.revision);
    if(!ui.dirty && result.validation) renderValidation(result.validation);
    return result;
  });
  return ui.chain;
}
function visibleCases() {
  const cohort=$("cohort").value, search=$("search").value.trim().toLowerCase(), status=$("status-filter").value;
  return ui.boot.cases.filter(r=>(cohort==="all" || (cohort==="pilot"?r.pilot:r.split===cohort))&&(!search||r.id.toLowerCase().includes(search))&&(status==="all"||(status==="unfinished"?r.status!=="completed":r.status===status)));
}
function renderQueue() {
  const rows=visibleCases();$("queue").replaceChildren();
  rows.forEach((row,i)=>{
    const button=el("button",undefined,"queue-item"+(row.id===ui.data?.id?" active":""));button.dataset.id=row.id;
    button.append(el("span",String(i+1).padStart(2,"0"),"queue-index"));const label=el("span",undefined,"queue-label");
    label.append(el("strong",row.id.replace("magicbrush:","#").replace(":"," · lượt ")),el("small",(row.split==="train"?"Train":"Validation")+" · "+statusNames[row.status]));
    button.append(label,el("span",row.status==="completed"?"✓":row.status==="deferred"?"—":"", "queue-status "+row.status));button.onclick=()=>loadCase(row.id).catch(showError);$("queue").append(button);
  });
  if(!rows.length) $("queue").append(el("p","Không có cặp phù hợp bộ lọc.","empty-state"));
  const i=rows.findIndex(r=>r.id===ui.data?.id);$("position").textContent=(i<0?"—":i+1)+" / "+rows.length;
}
function renderProgress() { const done=ui.boot.cases.filter(c=>c.status==="completed").length;$("total-progress").textContent=done+" / "+ui.boot.cases.length;$("total-bar").max=ui.boot.cases.length;$("total-bar").value=done; }
async function loadCase(id, force=false) {
  if(ui.navigating)return;ui.navigating=true;
  try {
  if(ui.data&&!force) {await save();if(ui.dirty) await save();}
  ui.busy=true;if(ui.state){renderGraphs();renderButtons();}
  clearTimeout(ui.timer);clearTimeout(ui.validationTimer);
  const data=await api("/api/case?id="+encodeURIComponent(id));
  ui.data=data;ui.state=clone(data.state);ui.phase=data.state.source_locked?1:0;ui.dirty=false;ui.conflict=false;ui.version++;ui.undo=[];ui.redo=[];ui.lastGroup=null;ui.selected={source:null,target:null};
  if(!ui.state.annotator) ui.state.annotator=localStorage.getItem(profileKey)||"";
  $("annotator").value=ui.state.annotator;notice("");
  ui.busy=false;
  renderAll();saved(data.state.revision?"Đã lưu vào máy · lượt "+data.state.revision:"Chưa có nhãn · bắt đầu từ ảnh nguồn");
  let local;
  try {local=JSON.parse(localStorage.getItem(localKey())||"null");} catch {notice("Bản nháp trình duyệt không đọc được. Dữ liệu trên máy chủ vẫn được giữ.",true);}
  if(local) {
    ui.conflict=true;renderGraphs();renderButtons();
    notice("Có bản nháp trong trình duyệt chưa được xác nhận lưu. Chọn cách khôi phục trước khi tiếp tục.");
    const backup=el("button","Tải bản nháp này","small-button");backup.onclick=()=>download("magicbrush-local-draft.json",local);
    const useServer=el("button","Dùng bản trên máy chủ","small-button");useServer.onclick=()=>{if(!confirm("Bỏ bản nháp trong trình duyệt và dùng bản đã lưu trên máy chủ? Có thể tải bản nháp trước khi bỏ."))return;localStorage.removeItem(localKey());ui.conflict=false;notice("");renderAll();};
    $("notice").append(document.createTextNode(" "),backup,document.createTextNode(" "),useServer);
    if(local.base_revision===ui.state.revision && local.source_locked===ui.state.source_locked) {
      const restore=el("button","Khôi phục bản nháp","small-button");restore.onclick=()=>{ui.conflict=false;for(const key of Object.keys(snapshot()))ui.state[key]=clone(local[key]);ui.state.annotator=local.annotator;$("annotator").value=local.annotator;ui.dirty=true;ui.version++;notice("");renderAll();save().catch(showError);};$("notice").append(document.createTextNode(" "),restore);
    }
  }
  } finally {ui.busy=false;ui.navigating=false;if(ui.state)renderButtons();}
}
function renderAll() {
  $("pair-meta").textContent=(ui.data.split==="train"?"TRAIN":"VALIDATION")+" / "+ui.data.id.replace("magicbrush:","CẶP ");
  $("notes").value=ui.state.notes;$("instruction-matches").checked=ui.state.instruction_matches;$("unchanged-checked").checked=ui.state.unchanged_checked;
  $("source-image").src="/api/image?id="+encodeURIComponent(ui.data.id)+"&role=source";
  if(ui.state.source_locked) {$("target-image").src="/api/image?id="+encodeURIComponent(ui.data.id)+"&role=target";$("instruction").textContent=ui.data.instruction;}
  else {$("target-image").removeAttribute("src");$("instruction").textContent="";}
  renderPhase();renderGraphs();renderQueue();renderProgress();renderButtons();
  if(ui.data.validation) renderValidation(ui.data.validation);
}
function renderPhase() {
  const locked=ui.state.source_locked;
  $("source-guide").hidden=locked;$("instruction-card").hidden=!locked;$("target-card").hidden=!locked;$("onboarding").hidden=locked;
  $("editor-grid").classList.toggle("pair",locked);$("check-card").hidden=ui.phase!==2;
  $("source-lock").textContent=locked?"Đã xác nhận":"Đang gán";$("source-readonly-note").hidden=!locked;
  $("step-target").disabled=!locked;$("step-check").disabled=!locked;$("reopen-source").hidden=!locked;
  for(const [i,key] of ["source","target","check"].entries()) {$("step-"+key).classList.toggle("active",ui.phase===i);$("step-"+key).classList.toggle("done",locked&&i===0);}
  $("work-title").textContent=ui.state.status==="completed"?"Cặp này đã hoàn thành":ui.phase===0?"Mô tả ảnh nguồn":ui.phase===1?"Sửa những gì khác ở ảnh đích":"Đối chiếu trước khi hoàn thành";
  $("work-subtitle").textContent=ui.phase===0?"Bắt đầu từ những đối tượng nổi bật và mối liên hệ giữa chúng.":ui.phase===1?"Graph nguồn đã được sao chép. Cùng một vật thể thì giữ nguyên dòng.":"Thao tác được tính từ hai graph. Bạn kiểm lại ý nghĩa theo hai ảnh.";
}
function renderButtons() {
  const disabled=ui.busy||ui.conflict;
  $("undo").disabled=disabled||!ui.undo.length;$("redo").disabled=disabled||!ui.redo.length;
  for(const id of ["primary","save","defer","reopen-source","export","backup","restore","previous","next"]) $(id).disabled=disabled;
  for(const id of ["annotator","notes","instruction-matches","unchanged-checked","source-quick-add"]) $(id).disabled=disabled;
  $("primary").textContent=ui.state.source_locked?(ui.phase===2?"Hoàn thành & cặp tiếp →":"Xem thay đổi & kiểm tra →"):"Xác nhận nguồn →";
  if(ui.state.status==="completed") $("primary").textContent="Cặp chưa xong tiếp theo →";
}
function renderGraphs() {renderGraph("source");if(ui.state.source_locked)renderGraph("target");}
function renderGraph(role) {
  const g=graph(role), edit=editable(role), list=$(role+"-nodes"), edges=$(role+"-edges");
  $(role+"-card").classList.toggle("readonly",!edit);list.replaceChildren();edges.replaceChildren();
  $(role+"-node-count").textContent=g.nodes.length;$(role+"-edge-count").textContent=g.edges.length;
  $(role+"-add-node").disabled=!edit;$(role+"-add-edge").disabled=!edit||g.nodes.length<2;
  if(!g.nodes.length) list.append(el("div","Chưa có đối tượng. Bấm lên ảnh hoặc chọn “Thêm đối tượng”.","empty-state"));
  for(const node of g.nodes) {
    const row=el("div",undefined,"node-row"+(ui.selected[role]===node.id?" selected":""));row.dataset.id=node.id;
    const badge=el("button",number(node.id),"node-number");badge.title="Chọn vật thể để đặt điểm trên ảnh";badge.onclick=()=>{ui.selected[role]=node.id;renderGraphs();};
    const fields=el("div",undefined,"node-fields");
    for(const [key,title,placeholder,suggestion] of [["label","Tên đối tượng","Ví dụ: cup","object-suggestions"],["attributes","Thuộc tính","red, ceramic","attribute-suggestions"]]) {
      const label=el("label"), input=el("input");label.append(el("span",title));input.value=key==="label"?node.label:node.attributes.join(", ");input.placeholder=placeholder;input.setAttribute("list",suggestion);input.setAttribute("aria-label",title+" "+roleName(role)+" "+number(node.id));input.disabled=!edit;input.dataset.field=key;
      input.oninput=()=>changed(()=>{graph(role).nodes.find(n=>n.id===node.id)[key]=key==="label"?input.value:input.value.split(",").map(s=>s.trim()).filter(Boolean);},false,role+node.id+key);
      input.onchange=()=>{refreshOptions(role);renderPins(role);};label.append(input);fields.append(label);
    }
    const remove=el("button","×","remove");remove.title="Xóa đối tượng";remove.setAttribute("aria-label","Xóa đối tượng "+roleName(role)+" "+number(node.id));remove.disabled=!edit;remove.onclick=()=>deleteNode(role,node.id);
    const meta=el("div",node.id.startsWith("newn")?"Vật thể mới ở ảnh đích":role==="target"?"Giữ cùng vật thể "+number(node.id)+" ở ảnh nguồn":ui.state[role+"_points"][node.id]?"Đã đánh dấu vị trí trên ảnh":"Chưa đánh dấu vị trí · không bắt buộc", "node-meta"+(node.id.startsWith("newn")?" new":""));
    row.append(badge,fields,remove,meta);list.append(row);
  }
  if(!g.edges.length) edges.append(el("div",g.nodes.length<2?"Thêm ít nhất hai đối tượng để nối quan hệ.":"Chỉ thêm quan hệ nhìn thấy rõ giữa các vật thể.","empty-state"));
  for(const edge of g.edges) {
    const row=el("div",undefined,"edge-row");row.dataset.id=edge.id;
    for(const [key,title] of [["subject","Từ vật thể"],["predicate","Quan hệ →"],["object","Đến vật thể"]]) {
      const label=el("label");label.append(el("span",title));let input;
      if(key==="predicate") {input=el("input");input.value=edge.predicate;input.placeholder="holding";input.setAttribute("list","relation-suggestions");input.oninput=()=>changed(()=>{graph(role).edges.find(e=>e.id===edge.id).predicate=input.value;},false,role+edge.id+key);}
      else {input=el("select");input.dataset.endpoint=key;fillOptions(input,g,edge[key]);input.onchange=()=>changed(()=>{graph(role).edges.find(e=>e.id===edge.id)[key]=input.value;},false);}
      input.setAttribute("aria-label",title+" "+roleName(role)+" "+edge.id);input.disabled=!edit;label.append(input);row.append(label);
    }
    const remove=el("button","×","remove");remove.setAttribute("aria-label","Xóa quan hệ "+roleName(role)+" "+edge.id);remove.disabled=!edit;remove.onclick=()=>changed(()=>{graph(role).edges=graph(role).edges.filter(e=>e.id!==edge.id);});row.append(remove);edges.append(row);
  }
  renderPins(role);
  if(role==="source") {
    const separate=!ui.state.source_locked&&g.nodes.length>0;
    $("source-editor").hidden=!separate;$("onboarding").hidden=ui.state.source_locked||separate;
    $("source-card").classList.toggle("sticky-source",separate);
    if(separate) $("source-editor").append($("source-form"));
    else $("source-card").insertBefore($("source-form"),$("source-readonly-note"));
  }
}
function fillOptions(select,g,value) {select.replaceChildren();for(const node of g.nodes){const o=el("option",number(node.id)+" · "+(node.label||"chưa đặt tên"));o.value=node.id;select.append(o);}select.value=value;}
function refreshOptions(role) {for(const row of $(role+"-edges").children){const edge=graph(role).edges.find(e=>e.id===row.dataset.id);if(!edge)continue;for(const select of row.querySelectorAll("select"))fillOptions(select,graph(role),edge[select.dataset.endpoint]);}}
function renderPins(role) {
  const container=$(role+"-pins");container.replaceChildren();
  for(const node of graph(role).nodes) {const xy=ui.state[role+"_points"][node.id];if(!xy)continue;const pin=el("button",number(node.id),"pin"+(ui.selected[role]===node.id?" selected":""));pin.style.left=(xy.x*100)+"%";pin.style.top=(xy.y*100)+"%";pin.title=node.label||"Chưa đặt tên";pin.setAttribute("aria-label","Điểm "+roleName(role)+" "+number(node.id));pin.onclick=e=>{e.stopPropagation();ui.selected[role]=node.id;renderGraphs();};container.append(pin);}
  if(role==="source"&&ui.state.source_locked)$("source-caption").textContent="Nguồn đã khóa. Đối chiếu và giữ đúng đối tượng khi sửa ảnh đích.";
}
function newId(role,kind) {const prefix=(role==="source"?"":"new")+(kind==="nodes"?"n":"e");const ids=new Set(graph(role)[kind].map(n=>n.id));let i=1;while(ids.has(prefix+String(i).padStart(3,"0")))i++;return prefix+String(i).padStart(3,"0");}
function addNode(role,point=null) {
  if(!editable(role))return;const id=newId(role,"nodes");changed(()=>{graph(role).nodes.push({id,label:"",attributes:[]});ui.selected[role]=id;if(point)ui.state[role+"_points"][id]=point;});
  $(role+"-nodes").querySelector('[data-id="'+id+'"] input').focus();
}
function deleteNode(role,id) {
  const g=graph(role), incident=g.edges.filter(e=>e.subject===id||e.object===id);
  if(!confirm("Xóa đối tượng này"+(incident.length?" cùng "+incident.length+" quan hệ liên quan":"")+"? Có thể hoàn tác."))return;
  changed(()=>{g.nodes=g.nodes.filter(n=>n.id!==id);g.edges=g.edges.filter(e=>e.subject!==id&&e.object!==id);delete ui.state[role+"_points"][id];if(ui.selected[role]===id)ui.selected[role]=null;});
}
function addEdge(role) {if(!editable(role)||graph(role).nodes.length<2)return;const g=graph(role),id=newId(role,"edges");changed(()=>{g.edges.push({id,subject:g.nodes[0].id,predicate:"",object:g.nodes[1].id});});$(role+"-edges").querySelector('[data-id="'+id+'"] input').focus();}
function scheduleValidation() {if(!ui.state.source_locked)return;clearTimeout(ui.validationTimer);ui.validationTimer=setTimeout(validate,450);}
async function validate() {const version=ui.version,id=ui.data.id;try{const result=await api("/api/validate",{id,target_graph:graph("target")});if(version===ui.version&&id===ui.data.id)renderValidation(result);}catch(error){showError(error);}}
function renderValidation(result) {
  $("validation-badge").textContent=result.valid?"✓ Hợp lệ về cấu trúc":"Cần kiểm tra";$("validation-badge").className="pill "+(result.valid?"good":"warn");$("validation-problems").textContent=result.problems.join(" ");$("operations").replaceChildren();
  if(!result.operations?.length) $("operations").append(el("p","Chưa có thay đổi giữa hai graph. Sửa graph đích theo những gì thực sự khác trong ảnh.","muted"));
  const names={add_node:"Thêm đối tượng",delete_node:"Xóa đối tượng",replace_node:"Thay nhãn / thuộc tính",add_edge:"Thêm quan hệ",delete_edge:"Xóa quan hệ",replace_edge:"Thay quan hệ"};
  const entity = value => value===null?"":value.label?value.label+attrs(value):nodeName(value.subject)+" → "+value.predicate+" → "+nodeName(value.object);
  function nodeName(id) {const n=[...graph("target").nodes,...graph("source").nodes].find(n=>n.id===id);return n?.label||"vật "+number(id);}
  for(const op of result.operations||[]) {const row=el("div",undefined,"operation");row.append(el("span",op.kind.startsWith("add")?"+":op.kind.startsWith("delete")?"−":"↔","op-icon"));const body=el("div");body.append(el("strong",names[op.kind]),el("small",op.before&&op.after?entity(op.before)+" → "+entity(op.after):entity(op.before||op.after)));row.append(body);$("operations").append(row);}
}
async function action(name) {
  if(ui.busy||ui.conflict)return;ui.busy=true;renderGraphs();renderButtons();
  try {await save();if(ui.dirty)await save();await save(name);ui.undo=[];ui.redo=[];ui.selected={source:null,target:null};ui.phase=ui.state.source_locked?1:0;notice("");}
  finally {ui.busy=false;renderAll();}
}
async function nextCase(direction=1,unfinished=false) {
  const rows=visibleCases();if(!rows.length)return;const index=rows.findIndex(c=>c.id===ui.data.id);
  for(let offset=1;offset<=rows.length;offset++){const row=rows[(index+direction*offset+rows.length*2)%rows.length];if(!unfinished||row.status!=="completed"){await loadCase(row.id);window.scrollTo({top:0,behavior:"smooth"});return;}}
  notice("Bạn đã hoàn thành toàn bộ danh sách đang chọn. Có thể mở rộng danh sách ở thanh bên.");
}
async function primary() {
  if(ui.state.status==="completed"){await nextCase(1,true);return;}
  if(!ui.state.source_locked){await action("lock_source");return;}
  if(ui.phase!==2){await save();ui.phase=2;renderPhase();await validate();$("check-card").scrollIntoView({behavior:"smooth",block:"start"});renderButtons();return;}
  await action("complete");await nextCase(1,true);
}
function travel(redo=false) {
  if(ui.busy||ui.conflict)return;const from=redo?ui.redo:ui.undo,to=redo?ui.undo:ui.redo;if(!from.length)return;
  to.push(snapshot());const restored=from.pop();Object.assign(ui.state,restored);ui.dirty=true;ui.version++;ui.lastGroup=null;persistLocal();renderAll();scheduleValidation();clearTimeout(ui.timer);ui.timer=setTimeout(()=>save().catch(showError),700);
}
for(const role of ["source","target"]) {
  $(role+"-add-node").onclick=()=>addNode(role);$(role+"-add-edge").onclick=()=>addEdge(role);
  $(role+"-image").onclick=event=>{if(!editable(role))return;const rect=event.target.getBoundingClientRect(),point={x:(event.clientX-rect.left)/rect.width,y:(event.clientY-rect.top)/rect.height};const selected=ui.selected[role];if(selected&&graph(role).nodes.some(n=>n.id===selected))changed(()=>{ui.state[role+"_points"][selected]=point;});else addNode(role,point);};
}
$("source-quick-add").onclick=()=>addNode("source");
$("primary").onclick=()=>primary().catch(showError);$("save").onclick=()=>save().catch(showError);$("previous").onclick=()=>nextCase(-1).catch(showError);$("next").onclick=()=>nextCase(1).catch(showError);
$("undo").onclick=()=>travel();$("redo").onclick=()=>travel(true);
$("reopen-source").onclick=()=>{if(confirm("Sửa lại nguồn sẽ mở một lượt gán mới và xóa bản đích hiện tại khỏi lượt này. Lịch sử cũ vẫn được giữ. Tiếp tục?"))action("reopen_source").catch(showError);};
$("defer").onclick=()=>{const reason=prompt("Lý do tạm bỏ qua (ví dụ: ảnh không khớp yêu cầu, sửa toàn cảnh, không rõ đối tượng):",ui.state.reason||"");if(reason===null)return;changed(()=>{ui.state.reason=reason;},false);action("defer").then(()=>nextCase()).catch(showError);};
$("annotator").oninput=()=>{localStorage.setItem(profileKey,$("annotator").value);if(ui.state)changed(()=>{ui.state.annotator=$("annotator").value;},false,"annotator");};
$("notes").oninput=()=>changed(()=>{ui.state.notes=$("notes").value;},false,"notes");
for(const [id,key] of [["instruction-matches","instruction_matches"],["unchanged-checked","unchanged_checked"]])$(id).onchange=()=>changed(()=>{ui.state[key]=$(id).checked;},false);
for(const id of ["search","cohort","status-filter"])$(id).addEventListener(id==="search"?"input":"change",()=>{renderQueue();});
$("backup").onclick=()=>download("magicbrush-draft-"+ui.data.id.replaceAll(":","-")+".json",{workspace_sha256:ui.boot.workspace_sha256,id:ui.data.id,base_revision:ui.state.revision,source_locked:ui.state.source_locked,annotator:ui.state.annotator,...snapshot()});
$("restore").onclick=()=>$("restore-file").click();
$("restore-from-help").onclick=()=>{$("help").close();$("restore-file").click();};
$("restore-file").onchange=async()=>{
  try {
    const file=$("restore-file").files[0];if(!file)return;if(file.size>300000)throw new Error("File bản nháp quá lớn.");
    const backup=JSON.parse(await file.text());
    if(backup.workspace_sha256!==ui.boot.workspace_sha256||!ui.boot.cases.some(c=>c.id===backup.id))throw new Error("Bản nháp thuộc bộ dữ liệu khác.");
    if(ui.conflict)throw new Error("Hãy xử lý bản nháp đang xung đột trước khi nhập file khác.");
    await save();if(ui.dirty)await save();
    const current=await api("/api/case?id="+encodeURIComponent(backup.id));
    if(current.state.revision!==backup.base_revision||current.state.source_locked!==backup.source_locked)throw new Error("Cặp này đã có bản lưu mới hơn. Không tự ghi đè bằng bản nháp cũ; hãy đối chiếu hoặc mở lượt gán mới.");
    if(!confirm("Khôi phục bản nháp cho "+backup.id+"? Bản hiện tại được giữ trong lịch sử."))return;
    await api("/api/save",{id:backup.id,action:"draft",payload:backup});
    localStorage.removeItem("magicbrush-manual:"+ui.boot.workspace_sha256+":"+backup.id);
    await loadCase(backup.id,true);notice("Đã khôi phục bản nháp. Hãy tiếp tục đối chiếu ảnh trước khi hoàn thành.");
  }catch(error){showError(error);}finally{$("restore-file").value="";}
};
$("export").onclick=async()=>{try{await save();const bundle=await api("/api/export",{});download("magicbrush-manual-"+bundle.report.snapshot.slice(0,12)+".json",bundle);notice("Đã xuất "+bundle.report.completed+" cặp hoàn thành. Nhãn vẫn chờ kiểm tra độc lập. Snapshot lưu tại "+bundle.report.directory);}catch(e){showError(e);}};
for(const [id,phase] of [["step-source",0],["step-target",1],["step-check",2]])$(id).onclick=()=>{ui.phase=phase;renderPhase();if(phase===2)validate();};
$("help-open").onclick=()=>$("help").showModal();$("help-close").onclick=()=>$("help").close();$("zoom-close").onclick=()=>$("zoom-dialog").close();
for(const button of document.querySelectorAll(".zoom"))button.onclick=()=>{$("zoom-image").src=$(button.dataset.role+"-image").src;$("zoom-dialog").showModal();};
document.addEventListener("keydown",event=>{
  if(!ui.state)return;const typing=["INPUT","TEXTAREA","SELECT"].includes(document.activeElement?.tagName);
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"){event.preventDefault();save().catch(showError);return;}
  if(document.querySelector("dialog[open]"))return;
  if(!typing&&(event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="z"){event.preventDefault();travel(event.shiftKey);return;}
  if(typing||event.ctrlKey||event.metaKey||event.altKey)return;
  const role=ui.state.source_locked?"target":"source";
  if(event.key.toLowerCase()==="n"){event.preventDefault();addNode(role);}
  else if(event.key.toLowerCase()==="r"){event.preventDefault();addEdge(role);}
  else if(event.key==="?")$("help").showModal();
  else if(event.key==="ArrowRight")nextCase(1).catch(showError);
  else if(event.key==="ArrowLeft")nextCase(-1).catch(showError);
  else if(event.key==="Escape"){ui.selected[role]=null;renderGraphs();}
});
window.addEventListener("beforeunload",event=>{if(ui.dirty){persistLocal();event.preventDefault();event.returnValue="";}});
async function start() {for(const id of ["primary","save","defer","reopen-source","export","backup","restore","previous","next","source-quick-add"])$(id).disabled=true;try{ui.boot=await api("/api/bootstrap");renderQueue();renderProgress();const first=visibleCases().find(c=>c.status!=="completed")||visibleCases()[0];if(first)await loadCase(first.id);else notice("Manifest chưa có cặp train/val để gán.");}catch(error){showError(error);}}
start();
