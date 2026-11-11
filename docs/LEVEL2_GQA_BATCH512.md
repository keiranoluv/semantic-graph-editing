# Full GQA v3 — batch 512/512

Theo yêu cầu người dùng, dừng v2 bằng SIGTERM và **train lại từ đầu** với
`train_batch_size=512`, `val_batch_size=512`. V2 đã lưu ở MLP seed 42,
epoch 5 / next_index 17024, hoàn tất 4 epoch; checkpoint/history giữ nguyên
trong `runs/level2/gqa-full-v2/`, W&B đánh dấu interrupted.

Config mới: [gqa-full-v3-bs512.json](../configs/level2/gqa-full-v3-bs512.json).
Output: `runs/level2/gqa-full-v3-bs512/`. Không nạp trọng số/optimizer của v2.
Giữ ma trận 15 lượt, max 200 epoch, patience 15/min_delta 0, AdamW lr 0,001,
weight decay 0,01, CLIP đóng băng và tiêu chí best validation GC mR@50.
Checkpoint mỗi **10 batch** (5.120 ảnh), thay vì 100 batch ở v2.

Cache đã hoàn tất ở `data/cache/gqa-full-clip-v1/` được dùng lại. Giữ
`clip_batch_size=64` để khớp identity cache; tham số này không còn tham gia bước
train. Không trích lại CLIP hoặc đổi nguồn/feature.

## Kiểm bộ nhớ trước khi khởi chạy

Đọc annotation train dự án, xét shuffle epoch đầu của cả ba seed rồi chọn batch
512 có nhiều candidate nhất và batch có nhiều sampled training pair nhất. Với
validation chọn batch có nhiều candidate nhất trong thứ tự cố định. Không dùng
test hoặc metric chất lượng để chọn. Mỗi model chạy train/validation trên các
batch này, 2 warmup + 3 lần đo, float32 trên RTX 5090; optimizer update là tạm
thời, không ghi checkpoint hoặc W&B. Cả **15 case pass**, loss/gradient hữu hạn.

| Model | Peak allocated train (GiB) | Peak allocated validation forward (GiB) |
|---|---:|---:|
| MLP | 1,399 | 1,723 |
| GAT | 4,660 | 4,506 |
| PE-CE | 7,382 | 2,587 |
| PE-PL | 7,789 | 2,587 |
| PE-PLPR | 7,789 | 2,587 |

Validation forward chưa gồm ranking/matcher CPU. Đây là số đo trên các batch
được chọn, **không phải cận bộ nhớ cho mọi cách ghép 512 graph** hoặc mọi epoch.
Cache loader dựng graph trên CPU vẫn là một phần chi phí đáng kể; không kỳ vọng
tăng batch 8 lần sẽ tăng throughput 8 lần. Full train còn 137 batch/epoch thay
vì 1.093, nên số optimizer update khác v2 dù cùng số epoch. So sánh chất lượng
phải ghi rõ recipe thay đổi; không gộp kết quả hai experiment.

[Benchmark JSON](../reports/level2/gqa-batch512-v1/benchmark.json) lưu batch
indices, seed, số pair, từng lần đo và source hash. Tái đo bằng:

```bash
myenv/bin/python -m tools.benchmark_gqa_batch512 \
  --output reports/level2/gqa-batch512-v2/benchmark.json
```

## Chạy / resume

Mặc định của launcher đã chuyển sang config v3. Lệnh tường minh:

```bash
scripts/run_gqa_full.sh --stage train \
  --config configs/level2/gqa-full-v3-bs512.json
```

Lượt được khởi động trong cửa sổ tmux **`0:gqa-bs512`**. Xem lại bằng
`tmux select-window -t 0:gqa-bs512` trong phiên hiện tại, hoặc `tmux attach -t 0`
từ terminal khác. Không chạy thêm cùng experiment khi cửa sổ này vẫn đang chạy.
Detach bằng Ctrl+B rồi D. W&B vẫn ở project `keiranoluv-hcmus/semantic-graph-editing`,
group mới theo fingerprint v3. Cách resume/early stopping/checkpoint giữ nguyên
[hướng dẫn v2](LEVEL2_GQA_FULL_V2_RUN.md).
