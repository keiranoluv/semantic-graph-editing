# Custom GQA: Edge-only và PE-CE + AlignG, v1

Hai biến thể bổ sung, không thay source/config/manifest của thí nghiệm cũ.
Không huấn luyện lại CLIP hoặc truy cập test.

| Tên launcher | Prototype update | Reverse feedback | Loss |
|---|---|---|---|
| edge-only | Tắt; normalized global P | Có | CE + PL + PR |
| pe-ce-aligng | GRU theo từng ảnh | Có | CE |

Trainer được version riêng từ aligng_fast_v1, chỉ thay factory và dispatch loss.
Vẫn giữ checkpoint nguyên tử, RNG sampler/dropout, resume, chọn epoch sớm nhất
khi đồng điểm và online W&B bắt buộc. CE có các loss PL/PR bằng 0.
Edge-only giữ module GRU trong state để schema/khởi tạo tương thích nhưng GRU
không được gọi và không nhận gradient. Tổng tham số vì vậy khác số tham số hoạt động.

## Recipe và queue

- Custom GQA: train 69.942, validation 5.000, 310 foreground predicate.
- Train/val/effective batch **512**, không gradient accumulation.
- Seed **42, 123, 3407**; tối đa **200 epoch**, early stopping **patience 15**.
- AdamW lr 0,001, weight decay 0,01, gradient clip 5; FP32, TF32 off.
- Chọn best theo validation GC mR@50; cuối lượt đánh giá GC và NG.
- Queue: **Edge42 → CE+AlignG42 → Edge123 → CE+AlignG123 → Edge3407 → CE+AlignG3407**.
- Chạy một model mỗi thời điểm; tự tiếp tục đủ sáu lượt. Lỗi/signal dừng queue;
  chạy lại cùng lệnh để resume lượt dang dở và bỏ qua lượt hoàn tất đã kiểm hash.
- Output: `runs/level2/aligng-ablation-v1/full/{model}/{seed}/`.
- W&B online: `keiranoluv-hcmus/semantic-graph-editing`; không fallback offline.

## Đối chứng đã có

Không chạy lại PE-CE, PE-PLPR hoặc PE-PLPR+AlignG. Preflight kiểm identity,
checkpoint hash và recipe của đủ 9 lượt control lịch sử:

- PE-CE: `runs/level2/gqa-full-v3-bs512/pe-ce/`.
- PE-PLPR và PE-PLPR+AlignG: `runs/level2/aligng-gqa-fast-v1/full/`.

So Edge-only với PE-PLPR+AlignG để kiểm GRU update; so với PE-PLPR để kiểm
reverse feedback. So PE-CE+AlignG với PE-CE để kiểm lợi ích khi không PL/PR;
đọc cùng cặp PLPR để đánh giá tương tác loss/feedback.
PE-CE lịch sử dùng engine PE-Net, cùng dữ liệu/recipe nhưng không khẳng định
dropout trajectory giống nhánh feedback mới. Chưa gọi đây là reproduction paper.

## Kiểm tra và chạy

```bash
# CPU plan, không train hoặc mở W&B
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m experiments.aligng_ablation_v1.runner --stage plan

# GPU capacity test sau commit: disposable steps, không lưu model train chính
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m experiments.aligng_ablation_v1.probe

# Người dùng chạy train chính sau khi capacity PASS
CUDA_VISIBLE_DEVICES=0 bash scripts/run_aligng_ablation_v1.sh

tail -f logs/aligng-ablation-v1.log
tmux attach -t aligng-ablation-v1
```

Probe chạy batch512 trên ba case lớn đã khóa, cho cả hai model; một warmup và
hai bước đo/case. Không W&B, không tạo best.pt/last.pt để dùng train chính.
Launcher sẽ từ chối train nếu thiếu/sai hash bằng chứng capacity.
Probe đã hoàn tất thì không chạy lại; dùng `--stage check` để kiểm readiness.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m experiments.aligng_ablation_v1.runner --stage check
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/aligng_ablation_v1 tests/aligng_fast_v1 tests/aligng_adapted_v1
```
