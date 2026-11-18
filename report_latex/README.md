# Báo cáo IEEE Conference — VG150, GQA, VN100 và video PVSG

Nguồn LaTeX theo mẫu Seminar ở `25C15039-25C15062/` (commit `5bdd556`).
Bản hiện tại gồm Nhiệm vụ 1 trên VG150, Task A của Nhiệm vụ 2 trên GQA, Nhiệm vụ 3 trên VN100 và phần mở rộng video trên PVSG.
Phụ lục A/B minh họa dự đoán trên validation, C bổ sung kết quả VN100 và D trình bày bảy lượt kiểm tra hội tụ muộn cùng mười timeline PVSG. Task B ở trạng thái TBD.

## Biên dịch

Từ thư mục này, với pdfLaTeX, TeX Live có phông T5/Babel tiếng Việt,
IEEEtran, TikZ, biblatex-ieee và Biber:

```sh
TEXINPUTS=.:../25C15039-25C15062/: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Gói ZIP để duyệt gồm các nguồn đang dùng, lớp IEEEtran của Seminar và
các hình. Trên Overleaf chọn `main.tex` và pdfLaTeX.

## Tái tạo hình và Bảng II

Mở **notebooks/vg150_report_figures.ipynb** từ gốc repo hoặc sau khi giải nén ZIP,
rồi chọn **Run All**. Notebook dùng Python 3.10+, Jupyter, pandas và Matplotlib
(đã chạy với Matplotlib 3.10.7). ZIP đi kèm các file JSON đầu vào đúng cấu trúc thư mục.

- Phần 1 hiển thị bảng đủ 50 predicate, gộp ngoài top 10 khi vẽ.
  Mẫu số gồm 405.860 annotation foreground từ 57.723 ảnh train, trước sampling.
- Phần 2 tính lại bốn metric và sample SD từ 54 lượt trên cùng 26.446 ảnh test,
  rồi đối chiếu với hai summary đã khóa.
- Phần 3 xuất các hàng điểm tuyệt đối tại K=100 cho Bảng II và vẽ chênh lệch
  tại K=50 sau khi ghép seed. Các cell hiển thị cả điểm và hiệu theo từng seed.

- Phần 4 ghép năm hàng hình H01–M01 đến H05–M05 từ các PNG từng ca và nhãn trong **cases.json**. Mỗi hàng được lưu thành một PDF riêng, với Hit bên trái và Miss bên phải.

Các tệp từ phần 1–3 là **vg-train-predicates.pdf**, **vg-ablation-deltas.pdf** và
**vg-ablation-at100.tex** trong thư mục figures của báo cáo. Phần 4 xuất **vg-pair-01.pdf** đến **vg-pair-05.pdf** vào cùng thư mục.

## Tái tạo bảng và hình trên GQA

Mở [notebook GQA](../notebooks/gqa_report_results.ipynb) trong môi trường của dự án
và chạy toàn bộ cell trên CPU. Notebook đối chiếu 27 kết quả trong
`reports/level2/gqa-test-v1/` và xuất `figures/gqa-test-rows.tex`. Phần hình
dùng cache validation, archive ảnh GQA và hai checkpoint PE-PLPR, PE-PLPR + AlignG
(seed 42) trong workspace để xuất `figures/gqa-validation-cases.pdf`.
Notebook đọc tần suất train từ `reports/level2/gqa-scale-audit-v1.json` để vẽ
`figures/gqa-train-predicates.pdf`. Sơ đồ `figures/gqa-penet-aligng.pdf` dùng
ảnh validation 1056, cùng annotation và dự đoán với ca Hit 4 trong phụ lục.

## Tái tạo ảnh minh họa VN100

Chạy [notebook Nhiệm vụ 3](../notebooks/level3_report_figures.ipynb) trên CPU trong workspace có ảnh pilot và VN100. Notebook dùng pdfLaTeX với phông tiếng Việt của báo cáo để xuất `nv3-sgg-pair.pdf`, `nv3-evaluation-cases.pdf` ở thân bài và hai hình `nv3-editor-pilot.pdf`, `nv3-color-case.pdf` ở Phụ lục C.

## Đối chiếu nội dung từ gốc repository

| Nội dung | Nguồn |
|---|---|
| Split, đặc trưng và bộ chấm | `scripts/build_vg150_manifest.py`, `docs/LEVEL1_MODEL_GUIDE.md`, `docs/evaluation/PredCls.md` |
| MLP, GAT, hai cách fusion | `src/models/mlp_semantic.py`, `src/models/gat.py`, `src/models/gat_pair_kg.py`, `src/models/gat_kg_fusion.py` |
| Hai bảng test GC và các hiệu số ghép seed | `reports/level1/reference-test27-v1/{summary,comparisons}.json` và `reports/level1/reference-loss-fusion-test27-v1/{summary,analysis}.json` |
| Long-tail và zero-shot | `reports/level1/test-tail-zero-v1/{long-tail,zero-shot}.md` |
| Hình mở đầu | Hình 1, trang 1, Zellers và cộng sự, *Neural Motifs*, CVPR 2018. Cắt từ bản PDF tác giả `1711.06640v2.pdf`; panel Input dùng vùng ảnh có bounding boxes. Panel Output minh họa một ground-truth triplet của cùng hình. |
| Ảnh định tính và kiểm tay | `reports/level1/submission/cases/{cases.json,analysis.md,human-review.csv,hit01..05.png,miss01..05.png}`. Biên bản chưa có xác nhận của người duyệt. |

## Hình và bảng PVSG

Chạy [notebook PVSG](../notebooks/pvsg_report_figures.ipynb) trên CPU với PyAV, NumPy và Matplotlib. Notebook dùng kết quả M5, video VidOR và cache PVSG v4 để vẽ phân bố predicate trên tập fit, minh họa hai cặp track qua nhiều frame, dựng timeline Hit/Miss và tính lại bảng test theo ba seed.
