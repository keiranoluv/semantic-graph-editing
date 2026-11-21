"use strict";
(() => {
  const packet = JSON.parse(document.getElementById("review-data").textContent);
  const classes = packet.classes;
  const byId = new Map(classes.map(row => [String(row.class_id), row]));
  const storageKey = `gqa-wordnet-review:${packet.packet_sha256}`;
  const $ = id => document.getElementById(id);
  let state = {reviews: {}, current: String(classes[0].class_id), defaultReviewer: ""};
  let storageAvailable = true;
  const cleanReview = () => ({decision: "", selected_synset: "", reviewer: "", notes: ""});
  const review = row => state.reviews[String(row.class_id)] || cleanReview();
  const message = (text, error = false) => {
    $("message").textContent = text;
    $("message").className = error ? "error" : "";
  };
  function errorsFor(row, item) {
    const errors = [];
    if (!["", "accept", "defer"].includes(item.decision)) errors.push("quyết định không hợp lệ");
    if (item.selected_synset && !row.candidates.some(c => c.synset_id === item.selected_synset)) errors.push("synset ngoài danh sách");
    if (item.decision && !item.reviewer.trim()) errors.push("thiếu tên người duyệt");
    if (item.decision === "accept" && !item.selected_synset) errors.push("accept cần chọn synset");
    if (item.decision === "accept" && !row.examples.length) errors.push("không có ảnh để accept");
    return errors;
  }
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (saved) {
      if (!saved.reviews || typeof saved.reviews !== "object" || Object.keys(saved.reviews).some(id => !byId.has(id))) throw new Error("draft IDs");
      for (const item of Object.values(saved.reviews)) {
        if (!item || ["decision", "selected_synset", "reviewer", "notes"].some(k => typeof item[k] !== "string")) throw new Error("draft fields");
      }
      state = {...state, ...saved};
      if (!byId.has(state.current)) state.current = String(classes[0].class_id);
    }
  } catch (_) {
    message("Không đọc được bản nháp cũ. Các quyết định hiện để trống; dùng CSV đã xuất để khôi phục.", true);
  }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(state)); }
    catch (_) { storageAvailable = false; message("Trình duyệt không lưu bản nháp. Hãy xuất CSV trước khi đóng trang.", true); }
  }
  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  };
  function isVisible(row) {
    const q = $("search").value.trim().toLowerCase();
    if (!`${row.class_id} ${row.label}`.toLowerCase().includes(q)) return false;
    const filter = $("filter").value;
    const decision = review(row).decision;
    return filter === "all" || filter === "pending" && !decision || filter === decision ||
      filter === "ambiguous" && row.status === "ambiguous" ||
      filter === "unsupported" && (!row.candidates.length || row.evidence_status !== "supported");
  }
  function renderNavigation() {
    const list = $("class-list");
    list.replaceChildren();
    let completed = 0, accepted = 0, deferred = 0;
    for (const row of classes) {
      const item = review(row), valid = !errorsFor(row, item).length;
      if (valid && item.decision) completed++;
      if (valid && item.decision === "accept") accepted++;
      if (valid && item.decision === "defer") deferred++;
      if (!isVisible(row)) continue;
      const button = element("button", undefined, `nav-item${String(row.class_id) === state.current ? " active" : ""}`);
      button.type = "button";
      button.dataset.classId = String(row.class_id);
      button.setAttribute("aria-current", String(row.class_id) === state.current ? "true" : "false");
      button.append(element("span", `${row.class_id} · ${row.label}`), element("small", !valid ? "Cần bổ sung" : item.decision || (row.status === "ambiguous" ? "Đa nghĩa" : !row.candidates.length ? "Chưa có synset" : "Chờ kiểm tra")));
      button.addEventListener("click", () => choose(row.class_id));
      list.append(button);
    }
    if (!list.children.length) list.append(element("p", "Không có nhãn khớp bộ lọc.", "empty"));
    $("progress").textContent = `${completed}/${classes.length} đã ghi quyết định · ${accepted} accept · ${deferred} defer`;
  }
  function choose(id) {
    state.current = String(id);
    persist();
    render();
  }
  function update() {
    const row = byId.get(state.current);
    state.reviews[state.current] = {
      decision: $("decision").value,
      selected_synset: document.querySelector("input[name=synset]:checked")?.value || "",
      reviewer: $("reviewer").value,
      notes: $("notes").value
    };
    persist();
    renderNavigation();
    const errors = errorsFor(row, review(row));
    $("validation").textContent = errors.length ? `Chưa thể xuất quyết định này: ${errors.join("; ")}.` : "";
  }
  function render() {
    const row = byId.get(state.current), item = review(row), index = classes.indexOf(row);
    $("position").textContent = `NHÃN ${index + 1} / ${classes.length} · CLASS ID ${row.class_id}`;
    $("class-title").textContent = row.label;
    const badges = $("badges"); badges.replaceChildren(element("span", row.status, `badge${["ambiguous", "unmatched"].includes(row.status) ? " caution" : ""}`));
    $("mapping-current").textContent = `Synset hiện tại: ${row.synset_id || "chưa liên kết"}`;
    $("mapping-definition").textContent = row.definition;
    const warnings = [];
    if (row.status === "linked_manual") warnings.push("Mapping kế thừa review VG150; chưa phải quyết định human review GQA.");
    if (row.status === "ambiguous") warnings.push("Nhãn đa nghĩa: cần chọn nghĩa phù hợp với các instance trong ảnh train, hoặc defer nếu các ảnh không thống nhất.");
    if (!row.candidates.length) warnings.push("Chưa có ứng viên WordNet. Accept bị tắt; dùng defer và ghi chú nghĩa cần bổ sung.");
    if (row.examples.length < 3) warnings.push(`Chỉ có ${row.examples.length} ảnh dùng được. Bằng chứng thiếu so với mục tiêu 3–5 ảnh; không tạo ảnh thay thế.`);
    if (row.image_errors.length) warnings.push(`${row.image_errors.length} ảnh không đọc được: ${row.image_errors.map(e => e.image_id).join(", ")}.`);
    $("class-warning").textContent = warnings.join(" ");
    $("evidence-count").textContent = `${row.examples.length} ảnh hiển thị / ${row.eligible_train_images} ảnh train có nhãn`;
    const examples = $("examples"); examples.replaceChildren();
    for (const example of row.examples) {
      const figure = element("figure", undefined, "example");
      for (const [kind, path] of [["context", example.context_path], ["crop", example.crop_path]]) {
        const link = element("a"); link.href = `../../${path}`; link.target = "_blank"; link.rel = "noopener";
        const image = element("img", undefined, kind); image.src = link.href;
        image.alt = `${row.label} · train image ${example.image_id} · object ${example.object_id}${kind === "context" ? " · khung đỏ" : " · crop"}`;
        image.loading = "lazy";
        image.addEventListener("error", () => { image.alt += " — không tải được ảnh"; figure.dataset.loadError = "true"; message(`Không tải được ảnh ${example.image_id}. Kiểm tra đường dẫn assets trước khi review.`, true); });
        link.append(image); figure.append(link);
      }
      figure.append(element("figcaption", `Train image ${example.image_id} · object ${example.object_id}\nBox annotation: [${example.annotation_box_xyxy.join(", ")}] · ${example.annotation_size.join("×")}\nJPEG: ${example.jpeg_size.join("×")}`));
      examples.append(figure);
    }
    const candidates = $("candidates"); candidates.replaceChildren();
    for (const candidate of row.candidates) {
      const label = element("label", undefined, "candidate"), radio = element("input");
      radio.type = "radio"; radio.name = "synset"; radio.value = candidate.synset_id;
      radio.checked = item.selected_synset === candidate.synset_id;
      radio.addEventListener("change", update);
      const text = element("span"), title = element("strong", candidate.synset_id);
      if (candidate.synset_id === row.synset_id) title.append(element("em", "mapping hiện tại"));
      text.append(title, document.createTextNode(candidate.definition)); label.append(radio, text); candidates.append(label);
    }
    if (!row.candidates.length) candidates.append(element("p", "Không có nghĩa danh từ trong mapping hiện tại. Ghi đề xuất bổ sung vào phần ghi chú.", "empty"));
    $("decision").value = item.decision;
    $("decision").querySelector("option[value=accept]").disabled = !row.candidates.length || !row.examples.length;
    $("reviewer").value = item.reviewer || state.defaultReviewer;
    $("notes").value = item.notes;
    $("previous").disabled = index === 0; $("next").disabled = index === classes.length - 1;
    $("validation").textContent = errorsFor(row, item).join("; ");
    renderNavigation();
  }
  function csvText() {
    const rows = [];
    for (const row of classes) {
      const item = review(row), errors = errorsFor(row, item);
      if (errors.length) throw new Error(`${row.label}: ${errors.join("; ")}`);
      rows.push({...row.original, ...item, train_image_ids: row.examples.map(e => e.image_id).join(";"), evidence_status: row.evidence_status, review_packet_sha256: packet.packet_sha256});
    }
    const quote = v => `"${String(v ?? "").replaceAll('"', '""')}"`;
    return "\ufeff" + [packet.export_fields, ...rows.map(row => packet.export_fields.map(key => row[key]))].map(row => row.map(quote).join(",")).join("\r\n") + "\r\n";
  }
  function parseCsv(text) {
    text = text.replace(/^\ufeff/, "");
    const rows = []; let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (c === '"') {
        if (quoted && text[i + 1] === '"') { field += '"'; i++; }
        else quoted = !quoted;
      } else if (c === "," && !quoted) { row.push(field); field = ""; }
      else if ((c === "\n" || c === "\r") && !quoted) {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += c;
    }
    if (quoted) throw new Error("CSV chưa đóng dấu ngoặc kép");
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows;
  }
  $("export-csv").addEventListener("click", () => {
    try {
      const text = csvText(), url = URL.createObjectURL(new Blob([text], {type: "text/csv;charset=utf-8"}));
      const link = element("a"); link.href = url; link.download = "gqa-wordnet-human-review.csv"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      message("Đã xuất CSV. Quyết định chưa điền vẫn để trống; chưa có bước áp dụng mapping.");
    } catch (error) { message(error.message, true); }
  });
  $("save-csv").disabled = location.protocol === "file:";
  $("save-csv").addEventListener("click", async () => {
    try {
      csvText();
      const response = await fetch("/api/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({packet_sha256: packet.packet_sha256, reviews: state.reviews})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Không lưu được CSV");
      message(`Đã lưu ${result.path}`);
    } catch (error) { message(error.message, true); }
  });
  $("import-csv").addEventListener("change", async event => {
    try {
      const file = event.target.files[0]; if (!file) return;
      const [header, ...rows] = parseCsv(await file.text());
      if (header.join("\0") !== packet.export_fields.join("\0") || rows.length !== classes.length) throw new Error("CSV không đúng schema hoặc số nhãn");
      const reviews = {};
      for (const values of rows) {
        if (values.length !== header.length) throw new Error("Sai số cột CSV");
        const record = Object.fromEntries(header.map((key, i) => [key, values[i]])), row = byId.get(record.class_id);
        if (!row || reviews[record.class_id] || record.label !== row.label || record.review_packet_sha256 !== packet.packet_sha256) throw new Error("CSV khác class ID, nhãn hoặc phiên bản review");
        for (const key of ["status", "synset_id", "definition", "candidates"]) if (record[key] !== row.original[key]) throw new Error(`Metadata gốc đã thay đổi: ${row.label}`);
        if (record.train_image_ids !== row.examples.map(e => e.image_id).join(";") || record.evidence_status !== row.evidence_status) throw new Error("Bằng chứng ảnh trong CSV không khớp");
        const item = {decision: record.decision, selected_synset: record.selected_synset, reviewer: record.reviewer, notes: record.notes};
        const errors = errorsFor(row, item); if (errors.length) throw new Error(`${row.label}: ${errors.join("; ")}`);
        reviews[record.class_id] = item;
      }
      state.reviews = reviews; persist(); render(); message("Đã nhập CSV đúng phiên bản review.");
    } catch (error) { message(error.message, true); }
    event.target.value = "";
  });
  $("default-reviewer").value = state.defaultReviewer;
  $("default-reviewer").addEventListener("input", () => { state.defaultReviewer = $("default-reviewer").value; persist(); if (!review(byId.get(state.current)).reviewer) $("reviewer").value = state.defaultReviewer; });
  for (const id of ["search", "filter"]) $(id).addEventListener("input", renderNavigation);
  for (const id of ["decision", "reviewer", "notes"]) $(id).addEventListener("input", update);
  $("clear-synset").addEventListener("click", () => { document.querySelectorAll("input[name=synset]").forEach(input => input.checked = false); update(); });
  $("previous").addEventListener("click", () => choose(classes[classes.indexOf(byId.get(state.current)) - 1].class_id));
  $("next").addEventListener("click", () => choose(classes[classes.indexOf(byId.get(state.current)) + 1].class_id));
  render();
  if (location.protocol === "file:" && storageAvailable) message("Chế độ file cục bộ: dùng Xuất CSV. Chạy CLI serve để lưu trực tiếp vào reviews/.");
})();
