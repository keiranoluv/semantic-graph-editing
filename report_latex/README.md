# Báo cáo IEEE Conference — VG150, GQA, VN100 và video PVSG

Nguồn LaTeX theo mẫu Seminar ở `25C15039-25C15062/` (commit `5bdd556`).
Bản hiện tại gồm Nhiệm vụ 1 trên VG150, Task A của Nhiệm vụ 2 trên GQA, Task B trên MagicBrush với đối chứng bổ sung trên phần crowdsourced của He và cộng sự (2020), Nhiệm vụ 3 trên VN100 và phần mở rộng video trên PVSG.
Phụ lục A/B minh họa dự đoán trên validation, C bổ sung kết quả VN100 và D trình bày bảy lượt kiểm tra hội tụ muộn cùng mười timeline PVSG. Phụ lục E tổng hợp các thử nghiệm Task B, VG150, ConceptNet và GQA.

## Biên dịch

Từ thư mục này, với pdfLaTeX, TeX Live có phông T5/Babel tiếng Việt,
IEEEtran, TikZ, biblatex-ieee và Biber:

```sh
TEXINPUTS=.:../25C15039-25C15062/: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Gói ZIP để duyệt gồm các nguồn đang dùng, lớp IEEEtran của Seminar và
các hình. Trên Overleaf chọn `main.tex` và pdfLaTeX.

## Hình 8 và 9 trên MagicBrush

Hình 8 dùng TikZ để vẽ các cạnh nối với bàn dưới cặp ảnh validation. Hình 9 dùng TikZ để đặt hai cặp ảnh test cạnh nhãn đích và dự đoán của seed 42. Mã vẽ và ảnh của hai ca test nằm trong [figures/pgf](figures/pgf/).

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
`reports/level2/gqa-test-v1/` và ba lượt GAT dùng adjusted loss trong
`reports/level2/gqa-gat-loss-test-v1/`, rồi xuất `figures/gqa-test-rows.tex`. Phần hình
dùng cache validation, archive ảnh GQA và hai checkpoint PE-PLPR, PE-PLPR + AlignG
(seed 42) trong workspace để xuất `figures/gqa-validation-cases.pdf`.
Notebook đọc tần suất train từ `reports/level2/gqa-scale-audit-v1.json` để vẽ
`figures/gqa-train-predicates.pdf`. Sơ đồ `figures/gqa-penet-aligng.pdf` dùng
ảnh validation 1056, cùng annotation và dự đoán với ca Hit 4 trong phụ lục.

## Đối chiếu Task B

Bộ dữ liệu đối chứng do He và cộng sự phát hành kèm bài [Scene Graph Modification Based on Natural Language Commands](https://aclanthology.org/2020.findings-emnlp.87/), đăng tại *Findings of EMNLP 2020*. Nhóm dùng phần crowdsourced trong [gói dữ liệu của tác giả](https://github.com/xlhex/SceneGraphModification#data). Ví dụ người đội mũ trong báo cáo được vẽ lại từ Hình 2 của bài gốc.

Nguồn cho số cặp, danh sách thao tác và phiếu 20 mẫu nằm trong [bộ dữ liệu MagicBrush theo prompt](../experiments/magicbrush_prompt_intent_v1/README.md). Hai ảnh minh họa lấy từ mẫu validation `magicbrush:484847:1`. [Kết quả MagicBrush](../reports/level2/magicbrush-prompt-intent-test-v1/README.md) ghi cách tái lập bảng test 59 cặp. [Kết quả trên phần crowdsourced của He và cộng sự (2020)](../reports/level2/emnlp2020-task-b-wandb-v4-test-v1/README.md) ghi các chỉ số của hai mô hình trên 530 mẫu test.

Từ gốc repo, chạy lệnh sau để dựng lại biểu đồ F1 của GNN ở Hình 11 và các bản Matplotlib của hai minh họa MagicBrush. Script dùng ảnh MagicBrush trong workspace, dự đoán test đã lưu của nhánh prompt ở seed 42 và bảng điểm trên phần crowdsourced.

```bash
myenv/bin/python -m experiments.magicbrush_prompt_intent_v1.report_figures
```

## Tái tạo ảnh minh họa VN100

Chạy [notebook Nhiệm vụ 3](../notebooks/level3_report_figures.ipynb) trên CPU trong workspace có ảnh pilot và VN100. Notebook dùng pdfLaTeX với phông tiếng Việt của báo cáo để xuất `nv3-sgg-pair.pdf`, `nv3-evaluation-cases.pdf` ở thân bài và hai hình `nv3-editor-pilot.pdf`, `nv3-color-case.pdf` ở Phụ lục C.

## Đối chiếu nội dung từ gốc repository

| Nội dung | Nguồn |
|---|---|
| Split, đặc trưng và bộ chấm | `scripts/build_vg150_manifest.py`, `docs/LEVEL1_MODEL_GUIDE.md`, `docs/evaluation/PredCls.md` |
| MLP, GAT, hai cách fusion | `src/models/mlp_semantic.py`, `src/models/gat.py`, `src/models/gat_pair_kg.py`, `src/models/gat_kg_fusion.py` |
| Hai bảng test GC và các hiệu số ghép seed | `reports/level1/reference-test27-v1/{summary,comparisons}.json` và `reports/level1/reference-loss-fusion-test27-v1/{summary,analysis}.json` |
| Hai hoán vị bổ sung cho pair fusion trong bảng VG150 | [Kết quả theo seed và đối chứng KG thật](../reports/level1/vg150-pair-permutations-v1/summary.json) |
| Long-tail và zero-shot | `reports/level1/test-tail-zero-v1/{long-tail,zero-shot}.md` |
| Hình mở đầu | Hình 1, trang 1, Zellers và cộng sự, *Neural Motifs*, CVPR 2018. Cắt từ bản PDF tác giả `1711.06640v2.pdf`; panel Input dùng vùng ảnh có bounding boxes. Panel Output minh họa một ground-truth triplet của cùng hình. |
| Ảnh định tính và kiểm tay | `reports/level1/submission/cases/{cases.json,analysis.md,human-review.csv,hit01..05.png,miss01..05.png}`. Biên bản chưa có xác nhận của người duyệt. |

## Hình và bảng PVSG

Chạy [notebook PVSG](../notebooks/pvsg_report_figures.ipynb) trên CPU với PyAV, NumPy và Matplotlib. Notebook dùng kết quả M5, video VidOR và cache PVSG v4 để vẽ phân bố predicate trên tập fit, minh họa hai cặp track qua nhiều frame, dựng timeline Hit/Miss và tính lại bảng test theo ba seed.

## Nguồn số liệu cho các thực nghiệm bổ sung

| Phép thử | Bằng chứng |
|---|---|
| Tạo nhãn và chọn cặp | [InstructPix2Pix](../reports/level2/instructpix2pix-caption-graph-v1/README.md), [grounded v2](../reports/level2/magicbrush-grounded-failure-audit-v1/README.md), [602 cặp MagicBrush](../reports/level2/magicbrush-manual-full602-v1/README.md) |
| REACT++ trên MagicBrush | [20 cặp](../reports/level2/magicbrush-react-pilot20-v1/cohort.json), [đối chiếu từng cặp](../reports/level2/magicbrush-react-pilot20-v1/sg-assessment.md), [kết quả ghép và thao tác](../reports/level2/magicbrush-react-pilot20-v1/pairs.json) |
| Giải mã và học thuộc | [v1](../reports/level2/magicbrush-manual-experiment-v1/failure-audit.json), [v2](../reports/level2/magicbrush-manual-experiment-v2/result-audit.json), [16 mẫu đầy đủ](../reports/level2/magicbrush-overfit16-v1/result-audit.json), [16 mẫu rút gọn](../reports/level2/magicbrush-compact-overfit16-v1/result-audit.json) |
| Chọn thực thể và nhãn thao tác | [Nhãn rút gọn](../reports/level2/magicbrush-compact-pilot-v1/result-audit.json), [liên kết từ](../reports/level2/magicbrush-lexical-pilot-v1/RESULTS.md), [sao chép tên](../reports/level2/magicbrush-easy-copy-v1/RESULTS.md), [pilot 14 mẫu](../reports/level2/magicbrush-prompt-pilot-v1/pilot.json) |
| EMNLP | [Ablation trên validation](../reports/level2/emnlp2020-task-b-gpu-v3/README.md), [baseline và đối chiếu v3/v4 trên test](../reports/level2/emnlp2020-task-b-wandb-v4-test-v1/README.md) |
| VG150 và ConceptNet | [GC/NG](../reports/level1/reference-comparison/comparison.md), [relation fusion](../docs/experiments/level1_fusion_v1.md), [TransE và RotatE](../docs/knowledge/rotate_conceptnet_workflow.md), [ComplEx](../docs/knowledge/complex_conceptnet_workflow.md) |
| Pilot GQA | [MLP và GAT](../reports/level2/GQA_PILOT.md), [PE-Net](../reports/level2/PENET_PILOT.md) |
