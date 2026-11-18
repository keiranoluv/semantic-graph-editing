# Bản nháp 1 — Nhiệm vụ 1 trên VG150

Nguồn LaTeX theo mẫu Seminar ở `25C15039-25C15062/` (commit `5bdd556`).
Bản nháp giữ Tóm tắt và Giới thiệu ở TBD. Phần đang duyệt là
`sections/vg.tex`, kèm `sections/vg_cases.tex` sau tài liệu tham khảo.
Các tệp LaTeX của những nhiệm vụ khác thuộc bản trước và chưa được đưa vào PDF này.

## Biên dịch

Từ thư mục này, với pdfLaTeX, TeX Live có phông T5/Babel tiếng Việt,
IEEEtran, TikZ, biblatex-ieee và Biber:

```sh
TEXINPUTS=.:../25C15039-25C15062/: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Gói ZIP để duyệt gồm các nguồn đang dùng, lớp IEEEtran của Seminar và
các hình. Trên Overleaf chọn `main.tex` và pdfLaTeX.

## Đối chiếu nội dung từ gốc repository

| Nội dung | Nguồn |
|---|---|
| Split, đặc trưng và bộ chấm | `scripts/build_vg150_manifest.py`, `docs/LEVEL1_MODEL_GUIDE.md`, `docs/evaluation/PredCls.md` |
| MLP, GAT, hai cách fusion | `src/models/mlp_semantic.py`, `src/models/gat.py`, `src/models/gat_pair_kg.py`, `src/models/gat_kg_fusion.py` |
| Hai bảng test GC và các hiệu số ghép seed | `reports/level1/reference-test27-v1/{summary,comparisons}.json` và `reports/level1/reference-loss-fusion-test27-v1/{summary,analysis}.json` |
| Long-tail và zero-shot | `reports/level1/test-tail-zero-v1/{long-tail,zero-shot}.md` |
| Hình mở đầu | Hình 1, trang 1, Zellers và cộng sự, *Neural Motifs*, CVPR 2018. Cắt từ bản PDF tác giả `1711.06640v2.pdf`; phần chữ và đồ thị giữ dạng vector. |
| Ảnh định tính và kiểm tay | `reports/level1/submission/cases/{cases.json,analysis.md,human-review.csv,hit03.png,miss03.png}`; hai contact sheet có trong `latex_report_source/figures/`. Biên bản chưa có xác nhận của người duyệt. |
