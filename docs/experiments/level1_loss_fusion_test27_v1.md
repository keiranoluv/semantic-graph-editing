# VG150 test: 12 loss + 15 fusion checkpoint

Từ gốc repository, kiểm CPU trước:

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.level1_loss_fusion_test27_v1 --dry-run
```

Chạy full sau commit:

```bash
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.level1_loss_fusion_test27_v1 --batch-size 128
```

- Loss: CE/adjusted × KG-off/on, 4 cấu hình × seeds 42/123/3407 = 12 checkpoint.
- Fusion: baseline, pair real/shuffled, visual real/shuffled, 5 × 3 = 15 checkpoint.
- Loại sáu relation-prior checkpoint có bất nhất L1/L2 đã audit.
- Cùng reference-v1 GC, 26.446 ảnh test, batch 128. Chỉ inference, không chọn lại epoch.
- Chạy tuần tự một model, tự chuyển lượt tiếp theo cho tới đủ 27 rồi dừng.
- Output `reports/level1/reference-loss-fusion-test27-v1/`: frozen plan, report từng seed,
  summary mean/sample std (đơn vị fraction 0–1).
- Kế thừa runner đã kiểm của lượt baseline; pin hash runner và hai source plan.
  Kiểm tất cả checkpoint và report có sẵn trước khi bắt đầu GPU.
- Bỏ qua report hợp lệ khi chạy lại; sai hash/protocol/split/cache hoặc JSON dở dang
  khiến launcher dừng, không ghi đè. Không chấm lại 27 baseline đã xong.
- Lock ngăn hai instance launcher này chạy đồng thời. Không chạy launcher GPU khác cùng lúc.
- Evaluation không yêu cầu W&B. Từng report lưu hash evaluator/checkpoint/dữ liệu/cache;
  summary lưu hash frozen plan và report.

Các checkpoint được chọn từ validation; tập test cũng đã được dùng đánh giá baseline.
Ghi rõ lịch sử này trong report; việc test đã từng được đánh giá không tự chứng minh
rằng ý tưởng cải tiến đã sử dụng test để lựa chọn thiết kế.

Chạy CPU tests: `CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q
tests/test_level1_loss_fusion_test27_v1.py tests/test_level1_reference_test27_v1.py`.

## NV1 · Gắn đúng vector ConceptNet có giúp pair fusion trên VG150?

Giữ nguyên GAT pair fusion, logit-adjusted loss (τ = 0,25) và ba seed train
42, 123, 3407. Mỗi hoán vị tráo vector giữa 138 lớp có liên kết ConceptNet và
giữ cố định ánh xạ trong suốt train và test. Nhóm dùng lại ba checkpoint KG
thật và ba checkpoint của hoán vị gốc, rồi train sáu lượt cho hai hoán vị mới.
Checkpoint được chọn theo validation mR@50. Cả sáu lượt mới đã được chấm
trên cùng 26.446 ảnh VG150 test bằng reference-v1 với graph constraint.
[Launcher và cấu hình](../../tools/vg150_pair_permutations_v1/__main__.py)

KG thật đạt trung bình **25,24% mR@50** và **62,37% R@50** qua ba seed.
Trong bảng, Δ là điểm KG thật trừ điểm KG tráo ở cùng seed, tính bằng điểm
phần trăm trước khi làm tròn.

| Hoán vị KG | mR@50 KG tráo (%) | Δ mR@50 seed 42 | Δ mR@50 seed 123 | Δ mR@50 seed 3407 | Δ mR@50 trung bình | Δ R@50 trung bình |
|---|---:|---:|---:|---:|---:|---:|
| 20261025 (gốc) | 25,00 | +0,40 | −0,82 | +1,14 | +0,24 | +0,02 |
| 20261026 | 24,81 | +0,48 | +0,003 | +0,80 | +0,43 | +0,14 |
| 20261027 | 24,92 | +0,36 | +0,10 | +0,51 | +0,32 | +0,35 |

Lợi thế trung bình của ánh xạ đúng lặp lại với cả hai hoán vị mới. Mức
tăng mR@50 nhỏ và phụ thuộc seed. Ở seed 123, KG thật thấp hơn hoán vị
gốc 0,82 điểm và gần như ngang hoán vị 20261026 (+0,003 điểm). Tám trong
chín hiệu số mR@50 dương, nhưng ba so sánh của mỗi seed dùng chung một
checkpoint KG thật. Đây là kiểm tra độ ổn định trên tập test đã xem trước
đó. [Kết quả từng seed và cả hai chỉ số](../../reports/level1/vg150-pair-permutations-v1/summary.json)
