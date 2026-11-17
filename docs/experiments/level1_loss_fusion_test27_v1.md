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
