"use strict";
(() => {
  const snapshot = JSON.parse(document.getElementById("confirmed-review-snapshot").textContent);
  const packet = JSON.parse(document.getElementById("review-data").textContent);
  if (snapshot.packet_sha256 !== packet.packet_sha256) throw new Error("Confirmed review belongs to a different packet");
  const key = `gqa-wordnet-review:${packet.packet_sha256}`;
  const receipt = `${key}:imported-snapshot:${snapshot.csv_sha256}`;
  const restore = () => {
    try {
      const previous = localStorage.getItem(key);
      const saved = previous && JSON.parse(previous);
      if (saved?.reviews && localStorage.getItem(receipt) === snapshot.csv_sha256) return;
      if (previous && !localStorage.getItem(`${key}:before-snapshot:${snapshot.csv_sha256}`)) {
        localStorage.setItem(`${key}:before-snapshot:${snapshot.csv_sha256}`, previous);
      }
    } catch (_) { /* The ordinary CSV importer also works without localStorage. */ }
    const input = document.getElementById("import-csv");
    const status = document.getElementById("message");
    const observer = new MutationObserver(() => {
      if (status.textContent === "Đã nhập CSV đúng phiên bản review.") {
        try { localStorage.setItem(receipt, snapshot.csv_sha256); } catch (_) { /* CSV remains available. */ }
        observer.disconnect();
      } else if (status.classList.contains("error")) {
        observer.disconnect(); // An invalid import must never be recorded as applied.
      }
    });
    observer.observe(status, {childList: true, characterData: true, subtree: true, attributes: true});
    const files = new DataTransfer();
    files.items.add(new File([snapshot.csv_text], snapshot.csv_name, {type: "text/csv;charset=utf-8"}));
    input.files = files.files;
    input.dispatchEvent(new Event("change", {bubbles: true}));
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", restore, {once: true});
  else restore();
})();
