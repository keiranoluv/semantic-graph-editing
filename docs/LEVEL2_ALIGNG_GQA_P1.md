# AlignG custom GQA — bàn giao train chính sau P1

P0 và P1 được giữ riêng. Không sửa `src/`, adapter P0, source/config/manifest
hoặc kết quả lịch sử. User đã cho phép assistant chạy GPU probe và pilot;
assistant chưa chạy full training hoặc VG150. Báo cáo và số liệu:
[aligng-gqa-p1-v1](../reports/level2/aligng-gqa-p1-v1/README.md).

## Cấu hình chính

- Dataset: custom GQA **69.942 train / 5.000 validation**, C=311 gồm background;
  giữ nguyên split, CLIP cache và GloVe. Không đọc held-out test annotations.
- Hai biến thể matched: `pe-plpr-aligng` và `pe-plpr` (feedback disabled).
  Cả hai có cùng initial tensor state và sampler policy. Baseline chứa
  module feedback không dùng để giữ checkpoint schema, không có gradient vào
  module đó; số parameter nền 2.467.109, adapter có thêm 922.368 parameter.
- Batch train **16 ảnh**, effective batch **16**, accumulation **1**; val batch
  **1 ảnh**. Chọn batch16 theo capacity probe trước khi thấy kết quả pilotbs16.
- Float32, D256/2heads/GRU, dropout0.2, checkpoint activation, pair chunk1024,
  attention query chunk64. Không AMP hoặc TF32.
- AdamW lr0.001, decay0.01, clip5; PE-Net PL/PR nguyên bản của repo:
  k1=10/k2=1/gamma1=1/gamma2=7, trọng số CE/PL/PR đều1.
- Seeds **42, 123, 3407**. Tối đa **200 epoch**, early stopping patience **15**,
  min_delta0; chọn **validation GC mR@50**, hòa giữ epoch sớm hơn.
- W&B **online bắt buộc**, entity/project
  `keiranoluv-hcmus/semantic-graph-editing`. Nếu không tạo được online run thì
  dừng trước optimizer; không fallback offline. Live online run chưa được
  mở trong P1; wiring/fail-closed đã được kiểm bằng CPU tests.
- Output `runs/level2/aligng-gqa-v1/full-bs16/{model}/{seed}/`.
- Mỗi lệnh chỉ **một model/seed**, không queue, không tự chạy seed/model kế.
  GPU có shared lock: nếu job khác giữ lock thì từ chối, không chờ nối tiếp.
- Train chính là thí nghiệm đánh giá phương pháp; pilot chưa chứng minh ưu
  thế AlignG. Không dùng kết quả full batch512 lịch sử làm matched control.

## Lệnh CPU xem kế hoạch

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage plan
```

## Lệnh GPU do user chạy, lần lượt từng dòng

Chạy từ `/workspace/workspace/semantic-graph-editing`. **Không chạy đồng thời
các dòng dưới.** Chờ job hiện tại kết thúc rồi chủ động chạy dòng tiếp theo.
Mỗi lần launcher kiểm hash cache/source và load graph vào RAM bằng 4 CPU workers.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr-aligng --seed 42
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr --seed 42
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr-aligng --seed 123
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr --seed 123
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr-aligng --seed 3407
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 myenv/bin/python -u -m experiments.aligng_gqa_v1.runner --config configs/level2/aligng-gqa-full-v1-bs16.json --stage train --model pe-plpr --seed 3407
```

## Checkpoint, theo dõi và resume

Launcher gate kiểm hai pilot batch1 hoàn tất, checkpoint hash đúng, initial
state và sampler traces giống nhau. Pilot batch16 được xác minh và báo cáo
riêng để kiểm cấu hình bàn giao. Gate không phụ thuộc phương pháp nào thắng.

`last.pt` atomic lưu AdamW, RNG CPU/CUDA/sampler, shuffle order, offset giữa
epoch, history và best model. Ctrl+C/SIGTERM lưu ở batch boundary rồi thoát130;
chạy lại **đúng cùng lệnh** để resume. Run hoàn tất được verify rồi skip,
không train lại. Không đổi config/source đang khóa để tiếp tục run cũ.
`best.pt` và GC/NG cuối phải tái tạo đúng best validation score.

W&B chỉ log epoch đã checkpoint bền vững; `wandb-latest.json` giữ URL và trạng
thái. `result.json` chứa số liệu, checkpoint hash, parameter count, epochs,
peak tensor CUDA allocation và sampling trace. Train chính cần cả ba seed
để tổng hợp mean/sample SD. VG150 và test evaluation là các bước sau, chưa
được launcher này triển khai hoặc chạy.
