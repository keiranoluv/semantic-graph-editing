# Tổng hợp EMNLP Task B — W&B v4, kết quả người dùng đã chạy

Ngày tổng hợp: **15/11/2026 (UTC+7)**. **Đủ 6/6 lượt chính**, MLP/GNN với seeds
42/43/44, W&B online. Phần audit validation trên **532 mẫu** được thực hiện bằng CPU, không inference.
Sau đó người dùng cho phép **eval GPU trên 530 test samples**; kết quả mới ở
[report test v4](../emnlp2020-task-b-wandb-v4-test-v1/README.md). Không có train thêm
hoặc W&B API trong các bước tổng hợp/eval này.

## Test GPU đã hoàn tất

Trung bình ± sample std qua ba seed, cùng 530 mẫu:

| Model | Micro F1 (%) | Exact operations/target (%) |
|---|---:|---:|
| MLP | 81.69 ± 0.31 | 67.74 ± 0.68 |
| GNN | 82.44 ± 0.39 | 68.55 ± 0.22 |

Đã chạy inference GPU thực tế trên đủ sáu checkpoint. Metrics và prediction
giống v3, phù hợp trọng số giống nhau. Xem report test cho P/R/F1 sáu loại,
baselines, OOV, cases và provenance. Test này đã từng được xem ở v3.

## Validation của best checkpoint

Trung bình ± sample standard deviation qua ba seed, đơn vị %. Mọi model/seed
được chấm trên cùng 532 mẫu/1.500 GT operations. Checkpoint chọn bằng minimum
validation token loss đúng protocol đã chốt; không chọn lại theo F1 sau khi xem
kết quả. Exact target graph bằng exact operation set trong lần đo này.

| Model | Micro F1 | Macro F1 (6 loại) | Exact operations/target | Valid transaction | Simple exact |
|---|---:|---:|---:|---:|---:|
| MLP | 80.39 ± 0.86 | 52.14 ± 1.45 | 66.04 ± 1.22 | 95.93 ± 1.13 | 25.48 ± 1.49 |
| GNN | 80.25 ± 0.57 | 51.72 ± 0.91 | 65.85 ± 1.10 | 96.05 ± 0.68 | 23.10 ± 2.89 |

Simple dùng proxy **đúng một GT operation**, 140 mẫu; không phải nhận định câu
lệnh đơn giản về ngôn ngữ. Chỉ số này thấp hơn exact toàn tập vì nhóm thao tác
khác nhau; không gộp hai mẫu số. 40/532 programs có payload OOV vẫn nằm trong
mẫu số; closed-vocabulary upper bound exact 92,48% không phải accuracy đạt được.

| Thao tác | GT support val | MLP P / R / F1 (%) | GNN P / R / F1 (%) |
|---|---:|---:|---:|
| add_node | 93 | 49.42 / 49.46 / 49.41 | 50.70 / 52.69 / 51.63 |
| delete_node | 357 | 95.03 / 96.45 / 95.74 | 95.25 / 97.39 / 96.31 |
| replace_node | 139 | 29.21 / 24.94 / 26.90 | 28.40 / 22.54 / 25.13 |
| add_edge | 188 | 45.03 / 44.68 / 44.82 | 40.50 / 41.49 / 40.95 |
| delete_edge | 715 | 95.18 / 96.74 / 95.95 | 95.26 / 97.39 / 96.31 |
| replace_edge | 8 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

[summary.json](summary.json) giữ per-seed metrics, TP/FP/FN và std từng loại.
Deletion chiếm **1.072/1.500 GT operations (71,47%)**, giải thích vì sao micro
F1 khoảng 80% trong khi macro chỉ khoảng 52%. Replace-edge có 8 cung thuộc
4 mẫu, TP=0 ở mọi seed: **support ít, F1=0**, không phải thiếu support/undefined.

MLP hơn GNN khoảng **0,14 điểm micro-F1**, nhỏ so với độ lệch giữa các seed;
chưa có bằng chứng GNN tốt hơn. GNN nhỉnh ở add-node/delete nhưng thấp hơn ở
replace-node/add-edge. Source graphs chỉ 1–3 node; MLP vẫn có edge endpoints,
nên so sánh này đo phần truyền thông điệp, không phải bỏ toàn bộ thông tin graph.

## Early stopping và độ tái lập

Config: batch 128, max 20/min 6 epochs, patience 3, min_delta 0,001, AdamW
LR 0,001/weight decay 0,01, hidden 96, GNN 2 layers, FP32/TF32 off.
Cả sáu lượt đều dừng vì early stopping, không chạm max epochs.

| Model/seed | Epoch dừng | Best epoch | Best val loss | Val micro F1 (%) | W&B |
|---|---:|---:|---:|---:|---|
| MLP/42 | 9 | 6 | 0.365690 | 79.55 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/6e6f2c029a7397f6dcfe) |
| MLP/43 | 11 | 8 | 0.350570 | 81.28 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/8bca5d7a4c24dd08153c) |
| MLP/44 | 11 | 8 | 0.367490 | 80.33 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/fad1f5962b9205a7a220) |
| GNN/42 | 11 | 8 | 0.375280 | 80.08 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/7ef84ff382c172acc236) |
| GNN/43 | 10 | 7 | 0.354538 | 80.89 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/eac6f27864f8a2154963) |
| GNN/44 | 9 | 6 | 0.373459 | 79.79 | [run](https://wandb.ai/keiranoluv-hcmus/semantic-graph-editing/runs/e17fede14c28a9e12584) |

V4 chạy **61 epoch tổng cộng**, so với 72 của v3: giảm **11 epoch (15,28%)**.
So với budget v4 tối đa 120 epoch là giảm 49,17%. Đây là giảm số epoch, không
quy đổi thành speedup: v4 còn decode validation mỗi epoch và có W&B.
`runtime_seconds` nội bộ cũng không bao gồm toàn bộ thời gian W&B init/finish.

Đã so sánh từng tensor của best.pt: **6/6 bộ tham số giống hệt v3**. File
prediction validation cũng giống byte-for-byte, mọi validation metric bằng nhau.
Như vậy v4 tái lập được v3 từ đầu, bổ sung tracking/resume/early stopping và
tránh các epoch sau overfit; không có bằng chứng tăng accuracy. Epoch dừng và
best epoch khớp replay được công bố trước khi người dùng chạy.

## Cách đọc W&B

Cả sáu `wandb-run.json` ghi mode online, đúng deterministic run ID và đúng
freeze/result. Sáu local W&B summaries khớp toàn bộ `selected_val/*`, best epoch,
best loss, epoch dừng và stopping reason; launcher log ghi đủ sáu lần sync.
Đây là kiểm chứng bằng local artifact/log, không gọi remote API để xác nhận
trạng thái server hiện tại.

**Dùng `selected_val/*` để báo cáo best checkpoint.** `val/*` trong summary là
metric epoch cuối, có thể khác best.pt. Ví dụ MLP/43: best epoch 8 có F1
81,28%, còn epoch cuối 11 có F1 82,03%; checkpoint vẫn là epoch 8 vì monitor
đã chốt là val loss, không phải max F1. Không chọn lại model/epoch để nâng điểm.
Các đường học đã xuất tại [learning-curves.csv](learning-curves.csv), đủ 61 row.

## Phân tích lỗi và giới hạn

Seed 42 được dùng làm điểm tham chiếu cố định, không chọn seed tốt nhất:

| Nhóm lỗi trên 532 mẫu | MLP/42 | GNN/42 |
|---|---:|---:|
| Đúng hoàn toàn | 345 | 346 |
| Sai entity identity | 43 | 50 |
| Sai tập loại thao tác | 35 | 33 |
| Sai payload | 81 | 78 |
| Parse/schema failure | 23 | 25 |
| Transaction failure | 5 | 0 |

Transaction hợp lệ khoảng 96% nhưng exact chỉ khoảng 66%: graph hợp lệ không
đồng nghĩa sửa đúng yêu cầu. Thay nhãn và gắn nhãn mới đúng entity còn yếu;
thêm node chưa bảo đảm thêm đúng quan hệ đi kèm. Trọng tâm cải thiện tiếp nên
là grounding/copy nhãn và quan hệ hiếm, thay vì chỉ tăng epoch. Đây là hướng
nghiên cứu đề xuất, không phải cấu hình mới đã được kiểm chứng.

[error-analysis.json](error-analysis.json) giữ taxonomy và 5 đúng/5 sai chọn
bằng hash ở từng run, không phân phối raw instruction/graph/payload. Full
predictions nằm cục bộ trong `runs/level2/emnlp2020-task-b-wandb-v4/`.
Phân loại này dựa trên GT graph đã chuyển đổi, không thay thế human review.

Các hạn chế dữ liệu vẫn giữ nguyên: graph release vô hướng, `att` biểu diễn
bằng node, correspondence cấu trúc bảo thủ thay vì ID lịch sử được phát hành,
source overlap bị loại theo fingerprint graph chứ không chứng minh image ID
khác nhau. Chưa có human review EMNLP mới; độ tin cậy ngữ nghĩa/agreement chưa
đo. Không suy toàn bộ yêu cầu cấp độ 2 của đề gốc đã hoàn thành từ kết quả này;
nguồn EMNLP thay MagicBrush/InstructPix2Pix vẫn phải được giải trình với giảng viên.

## Trạng thái test và artifact

`frozen-checkpoints.json` của train ghi `test_used=false`, `automatic_test=false`
vì đó là trạng thái lúc khóa train. Sau phép eval GPU của người dùng, test đã
được đo thật trong output riêng, không sửa artifact train.
[Report test v4](../emnlp2020-task-b-wandb-v4-test-v1/README.md) ghi rõ đây là
test đã từng được đánh giá ở v3. `summary.json` trong thư mục validation này
chỉ chứa validation; test metrics nằm trong report test riêng.

Code chạy chính được tạo ở commit **7af91ca** (runtime v4 từ `2821bbd`);
source/config/data/checkpoint đều được đối chiếu hashes. Xem
[provenance.json](provenance.json) và [validation.json](validation.json).
Không sửa source/config/manifest/receipts v2/v3/v4, MagicBrush hoặc notebook.

Chạy lại phần tổng hợp trên CPU, không W&B/GPU/inference:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.summarize_emnlp2020_task_b_wandb_v4_v1
```

Output thống kê bất biến ở `reports/level2/emnlp2020-task-b-wandb-v4-results-v1/`;
không ghi đè artifact train. Nếu nguồn thay đổi, tạo phiên bản báo cáo mới.
