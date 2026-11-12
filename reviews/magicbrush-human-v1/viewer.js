"use strict";
(() => {
  const packet = JSON.parse(document.getElementById("review-data").textContent), cases = packet.cases;
  const $ = id => document.getElementById(id), axes = packet.axes;
  const labels = {source_graph:"Graph nguồn đúng và đủ",target_graph:"Graph đích đúng và đủ",alignment:"ID object được căn chỉnh đúng",operations:"Thao tác đúng payload, không dư/thiếu",instruction_consistency:"Thao tác phản ánh đúng câu yêu cầu"};
  const key = `magicbrush-review:${packet.packet_sha256}`;
  let state = {current:0,reviews:{},defaultReviewer:""};
  const message = (text,error=false) => {$("message").textContent=text;$("message").className=error?"error":"";};
  try {const saved=JSON.parse(localStorage.getItem(key)||"null");if(saved && typeof saved.reviews==="object" && saved.current>=0 && saved.current<cases.length)state={...state,...saved};}
  catch(_){message("Không đọc được bản nháp. Hãy nhập CSV đã xuất nếu có.",true);}
  const blank=()=>Object.fromEntries(["decision",...axes,"reviewer","notes"].map(k=>[k,""]));
  const item=row=>state.reviews[row.id]||blank();
  const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
  const svg=(tag,attrs)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const[k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);return e;};
  const persist=()=>{try{localStorage.setItem(key,JSON.stringify(state));}catch(_){message("Trình duyệt không lưu được bản nháp; hãy xuất CSV trước khi đóng.",true);}};
  function errors(row,r){
    const values=["","correct","incorrect","uncertain"],problems=[];
    if(["decision",...axes].some(k=>!values.includes(r[k])))problems.push("lựa chọn không hợp lệ");
    if(r.decision){
      if(!row.graphs_ready)problems.push("chưa có graph để review");
      if(!r.reviewer.trim()||axes.some(k=>!r[k]))problems.push("cần tên người duyệt và đủ năm mục đánh giá");
      if(r.decision==="correct"&&axes.some(k=>r[k]!=="correct"))problems.push("kết luận Đúng chưa khớp các mục");
      if(r.decision==="incorrect"&&!axes.some(k=>r[k]==="incorrect"))problems.push("kết luận Sai cần chỉ ra mục sai");
    }return problems;
  }
  function navigation(){
    $("cases").replaceChildren();let completed=0;
    cases.forEach((row,index)=>{const r=item(row);if(r.decision&&!errors(row,r).length)completed++;
      const b=el("button",`${index+1}. ${row.img_id} / turn ${row.turn_index} · ${r.decision||(!row.graphs_ready?"chờ graph":"chưa kiểm")}`,index===state.current?"active":"");
      b.dataset.pairId=row.id;b.addEventListener("click",()=>{state.current=index;persist();render();});$("cases").append(b);});
    $("progress").textContent=`${completed}/${cases.length} đã kiểm · ${cases.filter(r=>r.graphs_ready).length} cặp có graph`;
  }
  function photograph(row,role){
    const target=$(role+"-photo"),meta=row[role+"_image"];target.replaceChildren();
    const link=el("a");link.href=`../../${meta.path}`;link.target="_blank";link.rel="noopener";
    const image=el("img");image.src=link.href;image.alt=`${role==="source"?"Nguồn":"Đích"} ${row.id}`;
    image.addEventListener("error",()=>message(`Không đọc được ảnh ${row.id}. Kiểm tra assets.`,true));link.append(image);target.append(link);
    if(row.graphs_ready){const overlay=svg("svg",{viewBox:"0 0 1000 1000",preserveAspectRatio:"none",class:"boxes"});
      for(const[id,b]of Object.entries(row.label[role+"_boxes"]||{})){overlay.append(svg("rect",{x:b[0],y:b[1],width:b[2]-b[0],height:b[3]-b[1]}));const t=svg("text",{x:b[0]+4,y:Math.max(25,b[1]+25)});t.textContent=id;overlay.append(t);}target.append(overlay);}
    $(role+"-meta").textContent=`${meta.width}×${meta.height} · SHA256 ${meta.sha256.slice(0,16)}…`;
  }
  function graph(container,raw){
    container.replaceChildren();if(!raw){container.append(el("p","Chưa có graph do model sinh. Không có đồ thị minh họa giả.","pending"));return;}
    const drawing=svg("svg",{viewBox:"0 0 480 280",class:"graph-svg",role:"img","aria-label":"Đồ thị dự đoán"}),positions={};
    raw.nodes.forEach((n,i)=>{const angle=2*Math.PI*i/Math.max(raw.nodes.length,1)-Math.PI/2;positions[n.id]=[240+170*Math.cos(angle),140+100*Math.sin(angle)];});
    for(const e of raw.edges){const a=positions[e.subject],b=positions[e.object];drawing.append(svg("line",{x1:a[0],y1:a[1],x2:b[0],y2:b[1]}));const t=svg("text",{x:(a[0]+b[0])/2,y:(a[1]+b[1])/2});t.textContent=e.predicate;drawing.append(t);}
    for(const n of raw.nodes){const[x,y]=positions[n.id];drawing.append(svg("circle",{cx:x,cy:y,r:19}));const t=svg("text",{x,y:y+4,"text-anchor":"middle"});t.textContent=n.id;drawing.append(t);}
    container.append(drawing);const nodes=el("ul",undefined,"nodes"),edges=el("ul",undefined,"edges");
    for(const n of raw.nodes)nodes.append(el("li",`${n.id}: ${n.label}${n.attributes.length?" ["+n.attributes.join(", ")+"]":""}`));
    for(const e of raw.edges)edges.append(el("li",`${e.id}: ${e.subject} — ${e.predicate} → ${e.object}`));
    container.append(nodes,edges);
  }
  function update(){const row=cases[state.current];state.reviews[row.id]={decision:$("decision").value,reviewer:$("reviewer").value,notes:$("notes").value,...Object.fromEntries(axes.map(k=>[k,$("axis-"+k).value]))};persist();navigation();$("validation").textContent=errors(row,item(row)).join("; ");}
  function render(){
    const row=cases[state.current],r=item(row);$("pair-id").textContent=`${row.id} · ${row.split} · ${row.source_group_id}`;$("instruction").textContent=row.instruction;
    $("status").textContent=!row.graphs_ready?"CHỜ SINH ĐỒ THỊ: ảnh thật đã sẵn sàng. Chưa thể chấm chất lượng graph/alignment/thao tác. Không tính đây là cặp nhãn đã hoàn thành.":`Nhãn tự động cần người kiểm. Trạng thái: ${row.label.status}. ${(row.label.problems||[]).join("; ")}`;
    photograph(row,"source");photograph(row,"target");graph($("source-graph"),row.label?.source_graph);graph($("target-graph"),row.label?.target_graph);
    $("operations").replaceChildren();const ops=row.label?.operations;
    if(!row.graphs_ready)$("operations").append(el("p","Chưa có thao tác; cần graph nguồn/đích thật trước khi căn chỉnh và diff.","pending"));
    else if(!ops.length)$("operations").append(el("p","Không có thay đổi được biểu diễn. Có thể graph bỏ sót chỉnh sửa hoặc yêu cầu ngoài schema.","warning"));
    else for(const op of ops){const p=el("div",undefined,"operation");p.append(el("strong",op.kind),el("span",`${JSON.stringify(op.before)} → ${JSON.stringify(op.after)}`));$("operations").append(p);}
    $("raw").textContent=JSON.stringify(row.label,null,2);$("ratings").replaceChildren();
    for(const axis of axes){const label=el("label",labels[axis]),select=el("select");select.id="axis-"+axis;
      for(const[v,t]of[["","Chưa chấm"],["correct","Đúng"],["incorrect","Sai"],["uncertain","Chưa đủ căn cứ"]]){const option=el("option",t);option.value=v;select.append(option);}select.value=r[axis];select.disabled=!row.graphs_ready;select.addEventListener("change",update);label.append(select);$("ratings").append(label);}
    $("decision").value=r.decision;$("decision").disabled=!row.graphs_ready;$("reviewer").value=r.reviewer||state.defaultReviewer;$("notes").value=r.notes;
    $("validation").textContent=errors(row,r).join("; ");$("previous").disabled=state.current===0;$("next").disabled=state.current===cases.length-1;$("position").textContent=`Cặp ${state.current+1}/${cases.length}`;navigation();
  }
  function exportText(){const data=cases.map(row=>{const r=item(row),issues=errors(row,r);if(issues.length)throw new Error(`${row.id}: ${issues.join("; ")}`);return{pair_id:row.id,img_id:row.img_id,turn_index:String(row.turn_index),split:row.split,source_sha256:row.source_image.sha256,target_sha256:row.target_image.sha256,evidence_sha256:row.evidence_sha256,...r,review_packet_sha256:packet.packet_sha256};});
    const quote=v=>`"${String(v??"").replaceAll('"','""')}"`;return"\ufeff"+[packet.export_fields,...data.map(r=>packet.export_fields.map(k=>r[k]))].map(r=>r.map(quote).join(",")).join("\r\n")+"\r\n";}
  function parseCsv(input){input=input.replace(/^\ufeff/,"");const rows=[];let row=[],field="",quoted=false;
    for(let i=0;i<input.length;i++){const c=input[i];if(c==='"'){if(quoted&&input[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
      else if(c===","&&!quoted){row.push(field);field="";}else if((c==="\r"||c==="\n")&&!quoted){if(c==="\r"&&input[i+1]==="\n")i++;row.push(field);rows.push(row);row=[];field="";}else field+=c;}
    if(quoted)throw new Error("CSV chưa đóng ngoặc kép");if(field||row.length){row.push(field);rows.push(row);}return rows;}
  $("export").addEventListener("click",()=>{try{const url=URL.createObjectURL(new Blob([exportText()],{type:"text/csv;charset=utf-8"})),link=el("a");link.href=url;link.download="magicbrush-human-review.csv";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);message("Đã xuất CSV; các đánh giá chưa điền vẫn để trống.");}catch(e){message(e.message,true);}});
  $("save").disabled=location.protocol==="file:";$("save").addEventListener("click",async()=>{try{exportText();const response=await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({packet_sha256:packet.packet_sha256,reviews:state.reviews})}),result=await response.json();if(!response.ok)throw new Error(result.error);message(`Đã lưu ${result.path}; ${result.report.reviewed_pairs}/${cases.length} cặp đã kiểm.`);}catch(e){message(e.message,true);}});
  $("import").addEventListener("change",async event=>{try{const file=event.target.files[0];if(!file)return;const[header,...rows]=parseCsv(await file.text());if(header.join("\0")!==packet.export_fields.join("\0")||rows.length!==cases.length)throw new Error("Sai schema hoặc số cặp CSV");const next={};
    for(const values of rows){if(values.length!==header.length)throw new Error("Sai số cột");const r=Object.fromEntries(header.map((k,i)=>[k,values[i]])),row=cases.find(x=>x.id===r.pair_id);if(!row||next[r.pair_id]||r.review_packet_sha256!==packet.packet_sha256||r.img_id!==row.img_id||r.turn_index!==String(row.turn_index)||r.split!==row.split||r.source_sha256!==row.source_image.sha256||r.target_sha256!==row.target_image.sha256||r.evidence_sha256!==row.evidence_sha256)throw new Error("CSV khác ID hoặc phiên bản bằng chứng");
      const review=Object.fromEntries(["decision",...axes,"reviewer","notes"].map(k=>[k,r[k]])),issues=errors(row,review);if(issues.length)throw new Error(issues.join("; "));next[row.id]=review;}state.reviews=next;persist();render();message("Đã nhập CSV cùng phiên bản bằng chứng.");}catch(e){message(e.message,true);}event.target.value="";});
  $("default-reviewer").value=state.defaultReviewer;$("default-reviewer").addEventListener("input",()=>{state.defaultReviewer=$("default-reviewer").value;persist();if(!item(cases[state.current]).reviewer)$("reviewer").value=state.defaultReviewer;});
  for(const id of["decision","reviewer","notes"])$(id).addEventListener("input",update);
  for(const[id,delta]of[["previous",-1],["next",1]])$(id).addEventListener("click",()=>{state.current+=delta;persist();render();});
  render();
})();
