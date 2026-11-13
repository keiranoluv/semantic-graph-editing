"use strict";
const packet = JSON.parse(document.getElementById("packet").textContent);
const $ = id => document.getElementById(id);
const key = "emnlp2020-review-" + packet.packet_sha256;
let reviews = JSON.parse(localStorage.getItem(key) || "{}"), index = 0;
const options = [["", "Chưa kiểm"], ["correct", "Đúng"], ["incorrect", "Sai"], ["uncertain", "Chưa rõ"]];
function fill(select) { options.forEach(([value, text]) => { const o = document.createElement("option"); o.value=value; o.textContent=text; select.append(o); }); }
packet.cases.forEach((r, i) => { const o = document.createElement("option"); o.value=i; o.textContent=r.id; $("cases").append(o); });
fill($("decision"));
packet.axes.forEach(axis => { const label=document.createElement("label"), select=document.createElement("select"); label.textContent=axis+" "; select.id=axis; fill(select); label.append(select); $("axes").append(label); select.onchange=capture; });
function capture() {
  const r = {decision: $("decision").value, reviewer: $("reviewer").value, notes: $("notes").value};
  packet.axes.forEach(a => r[a]=$(a).value); reviews[packet.cases[index].id]=r;
  localStorage.setItem(key, JSON.stringify(reviews)); progress();
}
function progress() { $("progress").textContent=Object.values(reviews).filter(r=>r.decision).length+" / "+packet.cases.length+" đã có kết luận"; }
function panel(title, graph) {
  const section=document.createElement("section"), h=document.createElement("h3"), pre=document.createElement("pre");
  h.textContent=title; pre.textContent=JSON.stringify(graph, null, 2); section.append(h, pre); $("graphs").append(section);
}
function show(i) {
  index=Math.max(0, Math.min(packet.cases.length-1, i)); $("cases").value=index;
  const row=packet.cases[index], label=row.label, r=reviews[row.id]||{};
  $("instruction").textContent=row.instruction;
  $("meta").textContent=row.id+" · official "+row.official_split+" · "+label.status+" · "+(label.problems.join(", ")||"qua kiểm cấu trúc; chờ kiểm ngữ nghĩa");
  $("graphs").replaceChildren();
  panel("Graph nguồn gốc", label.raw_source); panel("Graph đích gốc", label.raw_target);
  panel("Graph nguồn chuyển đổi", label.source_graph||label.independently_converted_source);
  panel("Graph đích chuyển đổi", label.target_graph||label.independently_converted_target);
  $("evidence").textContent=JSON.stringify({operations:label.operations, correspondence:label.alignment, validation:label.stages, lossy:label.lossy_conversion, attributes:label.attribute_analysis, problems:label.problems}, null, 2);
  $("decision").value=r.decision||""; $("notes").value=r.notes||""; $("reviewer").value=r.reviewer||$("reviewer").value;
  packet.axes.forEach(a=>$(a).value=r[a]||""); progress();
}
$("prev").onclick=()=>show(index-1); $("next").onclick=()=>show(index+1); $("cases").onchange=()=>show(Number($("cases").value));
$("decision").onchange=capture; $("notes").oninput=capture; $("reviewer").oninput=capture;
$("save").onclick=async()=>{
  capture();
  try { const response=await fetch("/api/save", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({packet_sha256:packet.packet_sha256,reviews})});
    const result=await response.json(); if(!response.ok) throw new Error(result.error);
    $("saved").textContent=result.path+" · ready="+result.report.ready;
  } catch(e) { $("saved").textContent=e.message; }
};
show(0);
