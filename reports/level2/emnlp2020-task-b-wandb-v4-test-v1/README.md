# EMNLP Task B v4 — kết quả eval GPU trên test, v1

Ngày tổng hợp: **15/11/2026 (UTC+7)**. Người dùng đã cho phép assistant dùng GPU
eval test. Code được kiểm trên CPU và commit **`de08d37` trước GPU**. Đã thực
hiện inference mới cho đủ **6/6 checkpoint v4** trên RTX 5090, FP32/TF32 off,
batch **128**, MLP/GNN seeds **42/43/44**, một model mỗi thời điểm.
Không train, chọn lại checkpoint hoặc init/sửa W&B trong eval này.

## Kết quả trên cùng 530 test samples

Đây là source-disjoint subset của official test 1.000 mẫu: 468 source overlap
train/val bị loại theo rule đã khóa, 2 correspondence mơ hồ. Cùng dataset receipt,
source groups, features, train-only vocabulary và evaluator với v3. Test đã từng
được xem ở v3; không mô tả lần này là đánh giá trên test mới chưa xem. Arguments
v4 được chọn từ history validation, không thay đổi từ kết quả test này.

Trung bình ± sample standard deviation qua 3 seed, đơn vị %. Exact operation
set và exact target graph bằng nhau. Simple là proxy một GT operation, 121 mẫu.

| Model | Micro F1 | Macro F1 (6 loại) | Exact operations/target | Simple exact | Valid transaction |
|---|---:|---:|---:|---:|---:|
| MLP | 81.69 ± 0.31 | 54.25 ± 0.51 | 67.74 ± 0.68 | 25.34 ± 1.91 | 95.60 ± 0.29 |
| GNN | 82.44 ± 0.39 | 55.03 ± 0.28 | 68.55 ± 0.22 | 24.79 ± 1.65 | 95.79 ± 0.39 |

| Thao tác | GT support test | MLP P / R / F1 (%) | GNN P / R / F1 (%) |
|---|---:|---:|---:|
| add_node | 105 | 58.53 / 60.95 / 59.72 | 62.50 / 65.40 / 63.86 |
| delete_node | 363 | 97.33 / 96.88 / 97.10 | 97.53 / 97.98 / 97.76 |
| replace_node | 121 | 30.57 / 25.34 / 27.71 | 30.59 / 24.79 / 27.38 |
| add_edge | 210 | 42.99 / 44.76 / 43.86 | 42.64 / 44.60 / 43.56 |
| delete_edge | 728 | 97.46 / 96.75 / 97.10 | 97.53 / 97.76 / 97.64 |
| replace_edge | 6 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

[metrics.json](metrics.json) giữ đủ TP/FP/FN, mỗi seed và std từng loại.
No-op baseline: F1=0%, exact=0%, valid=100%. Nearest training program bằng
CLIP instruction+pooled source: **F1 56,22%, exact 33,77%, valid 67,55%**;
cả hai baseline được tính lại trên CPU, không lấy từ kết quả cũ.

35/530 programs có OOV, vẫn giữ đầy đủ trong mẫu số; upper bound exact do
vocabulary là 93,40%, không phải điểm đã đạt. Replace-edge support 6 cung thuộc
3 mẫu, TP=0 ở cả hai model/mọi seed: F1=0, không phải undefined. Không thêm
hoặc sửa nhãn để bù support. Simple exact chỉ khoảng 25%, dù exact toàn tập
khoảng 68%; cần đọc cùng mẫu số và phân bố thao tác.

## Diễn giải cùng validation và v3

GNN hơn MLP 0,75 điểm micro-F1 trên test, nhưng validation MLP nhỉnh hơn
(80,39% so với 80,25%). Chưa đủ bằng chứng kết luận GNN tổng quát tốt hơn từ
ba seed và source graph 1–3 node. Điểm mạnh tập trung vào deletion; macro F1
chỉ 54–55%, thay nhãn node khoảng 27%, thay cạnh chưa thành công.

Sau khi chạy GPU thật, xác nhận **6/6 metrics và file prediction test giống
v3**. Đây là kết quả mới được đo, không phải chép metric test v3 thành v4.
Điều này khớp kiểm tra trước eval rằng mọi tensor best checkpoint v4 bằng v3.
V4 đóng góp tracking, resume và early stopping: 61 thay vì 72 epoch, giảm
15,28% số epoch mà giữ nguyên kết quả; không có accuracy gain so với v3.

Case IDs và taxonomy của 5 đúng/5 sai theo hash ở mỗi lượt nằm trong
[error-analysis.json](error-analysis.json). Vì output trùng v3, có thể đối chiếu
[phân tích định tính v3](../emnlp2020-task-b-gpu-v3/README.md): sai nhãn gần nghĩa,
sai node được thay, replace giữ nguyên payload và sai loại thay quan hệ.
Đây là đối chiếu với GT graph release, chưa phải human semantic review.

Hạn chế representation/correspondence, graph vô hướng và attribute node,
source grouping chưa chứng minh image-disjoint, ít support quan hệ hiếm và
việc thay nguồn MagicBrush/InstructPix2Pix vẫn giữ nguyên. Không dùng kết quả
này để kết luận mọi yêu cầu cấp độ 2 của đề gốc đã được đáp ứng.

## Provenance và kiểm chứng

Artifact train v4 giữ nguyên, gồm online W&B của cả sáu lượt. Test output riêng:
`runs/level2/emnlp2020-task-b-wandb-v4-test-v1/`. Trạng thái `test_used=false`
trong training freeze là ảnh chụp tại lúc khóa train; test evidence nằm riêng
ở [provenance.json](provenance.json) và [artifact-receipt.json](artifact-receipt.json).
Không có lượt train hoặc hàng đợi khác được khởi chạy sau eval.

13 test CPU pass trước commit/GPU, gồm audit barrier, không optimizer/không W&B,
reuse và chống sửa checkpoint. Các raw predictions/graphs/instructions không
commit; report chỉ chứa thống kê, IDs và hashes. Xem [validation.json](validation.json).

Lệnh đã chạy (stage eval chạy đủ 6 lượt tuần tự rồi dừng, không train):

```bash
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.eval_emnlp2020_task_b_wandb_v4_test_v1 --stage eval
```

Tổng hợp lại trên CPU từ artifact đã hoàn tất, không inference/W&B:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.eval_emnlp2020_task_b_wandb_v4_test_v1 --stage summarize
```

[Hướng dẫn eval](../../../docs/LEVEL2_EMNLP2020_TASK_B_V4_TEST_V1.md) ·
[Validation/early stopping/W&B v4](../emnlp2020-task-b-wandb-v4-results-v1/README.md).
