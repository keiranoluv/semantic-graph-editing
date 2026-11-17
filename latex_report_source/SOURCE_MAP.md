# Nguồn số liệu của bản báo cáo viết lại

Bản này cập nhật VG150 sang test reference-v1; không còn khóa toàn bộ báo cáo
ở commit 927bf29c. Đường dẫn dưới đây tính từ gốc repository.
Các checkpoint/source/manifest thí nghiệm được giữ nguyên.

| Nội dung | Nguồn chính | Split/phạm vi |
|---|---|---|
| Đối chiếu đề bài | Project01_Graph_Editing.pdf | Đề người dùng cung cấp |
| VG150 9 cấu hình gốc | reports/level1/reference-test27-v1/README.md, summary.json, audit.json, comparisons.json | Test 26.446 ảnh, 27 checkpoint; báo cáo commit 5afb96cf |
| Loss/fusion 9 cấu hình | reports/level1/reference-loss-fusion-test27-v1/tables.md, analysis.json, summary.json | Test 26.446 ảnh, 27 checkpoint; báo cáo commit 4d036a10 |
| Protocol PredCls | docs/evaluation/PredCls.md; docs/LEVEL1_REPORT.md | GC reference-v1; chọn checkpoint legacy cố định |
| Long-tail | reports/level1/test-tail-zero-v1/long-tail.json, long-tail.md | Test 26.446 ảnh, 54 checkpoint; CPU từ per-predicate có sẵn |
| Zero-shot | reports/level1/test-tail-zero-v1/zero-shot-summary.json, analysis.json | Test; 9 checkpoint fusion, 4.660 ảnh có 7.848 GT zero-shot; code trước GPU e28a95b1 |
| 5 hit / 5 miss | reports/level1/submission/cases/{analysis.md,cases.json,human-review.csv} | GAT seed 42, validation; human review còn trống |
| GQA 5 model | reports/level2/gqa-full-v3-results-v1/README.md | Custom GQA validation |
| AlignG matched | reports/level2/aligng-gqa-fast-v1-results-v1/{summary.json,comparison.json} | Custom GQA validation |
| Adapted methods | docs/LEVEL2_METHOD_DECISION.md | Không gọi reproduction |
| MagicBrush kết quả | reports/level2/magicbrush-manual-experiment-v3/result-audit.json | Test 69; artifact local hiện chưa được git track |
| MagicBrush chẩn đoán | reports/level2/magicbrush-manual-v3-error-analysis-v1/README.md | Validation; không ghép mẫu số test |
| EMNLP thiết lập | reports/level2/emnlp2020-task-b-wandb-v4-results-v1/README.md | Train/validation |
| EMNLP kết quả | reports/level2/emnlp2020-task-b-wandb-v4-test-v1/{README.md,metrics.json,error-analysis.json} | Test 530 mẫu; 3 seed |
| VN100 pipeline | docs/LEVEL3_PIPELINE.md; docs/LEVEL3_EVALUATION.md | Qwen + REACT++, scorer cố định |
| VN100 kết quả/case | docs/LEVEL3_RESULTS_V1.md; configs/level3/evaluation200-v1.json | 200 edit, tương quan trên tập giao 91 |

Các biểu đồ vector PGFPlots trong PDF lấy số hiển thị ở các bảng tương ứng:
delta test VG150, delta AlignG validation, khoảng bootstrap Task 3.
Mean/std ở bảng là phần trăm; delta là điểm phần trăm; Spearman không nhân 100.

Hai contact sheet VG150 được sao chép nguyên trạng từ artifact case, không sinh
ảnh hoặc sửa prediction. Nhận xét hình chưa được nhóm xác nhận kiểm tay.
