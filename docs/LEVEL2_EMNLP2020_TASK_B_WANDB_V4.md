# EMNLP Task B v4 — W&B và early stopping, người dùng chạy train chính

V4 giữ nguyên dataset, features, vocabulary, model và kết quả cũ. Không sửa
`src/`, source/manifest/config v2/v3, MagicBrush hoặc notebook cá nhân. Các
lượt train chính v4 do người dùng chạy; launcher dành cho người dùng. Đã có
[tổng hợp sáu lượt validation](../reports/level2/emnlp2020-task-b-wandb-v4-results-v1/README.md).

## Arguments trước và sau

| Argument | GPU v3 đã chạy | V4 bàn giao |
|---|---|---|
| Models | MLP, GNN | MLP, GNN |
| Seeds | 42, 43, 44 | 42, 43, 44 |
| Train/eval batch | 128 | 128 |
| Max epochs | 12 cố định | **20** |
| Minimum epochs | Không có | **6** |
| Early stopping | Không có | **patience 3 epoch, min_delta 0,001** |
| Monitor / checkpoint | Minimum val token loss | Giữ nguyên minimum val token loss |
| Validation | Loss mỗi epoch, decode cuối lượt | Loss + decoded metrics **mỗi epoch** |
| AdamW LR / weight decay | 0,001 / 0,01 | Giữ nguyên |
| Hidden / GNN layers / GRU byte dim | 96 / 2 / 48 | Giữ nguyên |
| Gradient clipping | 5 | 5 |
| Precision | FP32, TF32 tắt | Giữ nguyên |
| W&B | Chưa tích hợp | **Online bắt buộc cho main** |

Train/dev sạch giữ 29.868/532 mẫu; cùng CLIP text cache CPU đã khóa và
train-only vocabulary 4.762 token. Batch 128 đã chạy ổn trong v3; lần này không
đổi batch/LR/kiến trúc cùng lúc với stopping policy.

History val v3 có best epoch MLP=6/8/8, GNN=8/7/6; các epoch sau train loss
còn giảm nhưng val loss tăng. Vì vậy không tăng max epochs lớn để cố hội tụ
train. Max 20 tạo dư địa cho seed khác; minimum 6 ngăn dừng quá sớm; patience 3
cho phép một vài dao động validation. `min_delta=0,001` tính trên **token cross
entropy tuyệt đối**, không phải F1 hoặc phần trăm. Chỉ history train/val v3 được
dùng để chọn arguments, không đọc/tune theo test metrics.

Hai quyết định tách biệt: checkpoint ghi nhận **mọi minimum val loss thực tế**;
bộ đếm patience chỉ reset khi loss giảm hơn 0,001 so với mốc cải thiện đáng kể
gần nhất. Minimum epochs chỉ chặn quyết định dừng, không bỏ các epoch đầu khỏi
checkpoint selection. Không có scheduler hoặc class weighting mới.

## Lệnh train chính — người dùng chạy

Đã dùng môi trường `myenv`, `wandb==0.18.7` trong requirements hiện có.
Đăng nhập W&B nếu máy chưa đăng nhập:

```bash
cd /workspace/workspace/semantic-graph-editing
myenv/bin/wandb login
bash scripts/run_emnlp2020_task_b_wandb_v4.sh
```

Project mặc định **`semantic-graph-editing`**. Để chọn team/account, đặt entity
trước khi chạy, ví dụ `WANDB_ENTITY=ten-team bash scripts/run_emnlp2020_task_b_wandb_v4.sh`.
Không đặt API key vào config hoặc command lưu trong repo. Launcher đặt
`WANDB_MODE=online`; `wandb.init(mode="online", force=True)` phải trả về online
run trước gradient đầu tiên. Lỗi auth/init/log/finish dừng hàng đợi, không tự
fallback offline/disabled hoặc CPU. Run ID ổn định, `resume="allow"`, mỗi
model/seed một run và cùng group theo experiment identity. Cơ chế này theo
[SDK W&B v0.18.7](https://github.com/wandb/wandb/blob/v0.18.7/wandb/sdk/wandb_init.py).

Thứ tự tự động: **MLP/42 → MLP/43 → MLP/44 → GNN/42 → GNN/43 → GNN/44**.
Một model/seed mỗi thời điểm. Stage train **tự chạy đủ sáu lượt**, dừng khi
lỗi và **không tự chạy test, sweep hoặc lượt train khác**. Queue lock chặn hai
launcher v4 cùng chạy. Config ở `configs/level2/emnlp2020-task-b-wandb-v4.json`.
Không tiếp tục trọng số v3: main v4 khởi tạo từ đầu; rerun v4 tiếp tục chính run
v4 đó nếu bị ngắt.

Output chính: `runs/level2/emnlp2020-task-b-wandb-v4/`.
Log: `logs/emnlp2020-task-b-wandb-v4/launcher.log`.
Mỗi model/seed có `last.pt` (optimizer/RNG/early-stop/history/best state),
`best.pt`, `result.json`, `val-predictions.json`, `val-cases.json`,
`wandb-run.json` (URL) và completion receipt. `frozen-checkpoints.json` chỉ tạo
sau khi cả sáu lượt xong và ghi `automatic_test=false`.

W&B theo dõi mỗi epoch: train/val loss, LR, per-type P/R/F1/support/TP/FP/FN,
micro/macro F1, exact operation/target, simple exact, valid transaction,
decode/transaction errors, OOV count, best epoch/loss, bad epochs và GPU memory.
Summary ghi kết quả checkpoint đã chọn và stopping reason. Chỉ log config,
metrics và hashes/IDs; không gửi instruction, vocabulary, graph, raw dataset
hoặc checkpoint lên W&B. `save_code=False`, git capture tắt.

`last.pt` được lưu nguyên tử sau epoch, trước gửi log. Khi chạy lại cùng lệnh:
khôi phục model/optimizer/RNG/best/early-stop, tiếp tục từ epoch hoàn tất cuối;
epoch đang dở được chạy lại. Replay các history row chưa log; không thêm lại
các epoch W&B đã có. Lượt hoàn tất được kiểm hash và bỏ qua; thay commit chỉ
về báo cáo không làm thay creation commit đã lưu. Thay code/config/data phải
đổi output version; không sửa receipt để chạy đè.

## Preflight và smoke test — không W&B

```bash
# Chỉ CPU, kiểm toàn train/val và cache, không W&B, không train
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_wandb_v4 --stage preflight

# CPU smoke trước commit code GPU
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m tools.emnlp2020_task_b_wandb_v4 --stage smoke --device cpu

# GPU smoke sau CPU tests và commit; không W&B
CUDA_VISIBLE_DEVICES=0 myenv/bin/python -m tools.emnlp2020_task_b_wandb_v4 --stage smoke --device cuda
```

Smoke mỗi model 1 epoch, seed 42, 8 mẫu train +4 mẫu chẩn đoán lấy từ official
train và khác source group, batch cấu hình 128 (batch thực tế train=8/val=4).
Không đọc official val để chọn cấu hình từ smoke, không mở test. Output riêng
`runs/level2/emnlp2020-task-b-wandb-v4-smoke-{cpu,cuda}/`; không tính là main
experiment, không dùng để báo accuracy hoặc chọn model. Không import/init
W&B trong execution smoke. Chạy hai model tuần tự, không mở hàng đợi chính.

```bash
CUDA_VISIBLE_DEVICES='' myenv/bin/python -m pytest -q tests/editing/test_emnlp2020_task_b_wandb_v4.py
```

Test CPU dùng W&B fake để kiểm run IDs, online-only, lỗi logging, resume, per-type
metrics và early stopping; không truy cập W&B thật. GPU smoke không kiểm kết
nối tài khoản W&B; kết nối được kiểm khi người dùng khởi chạy main. V4 đã hoàn tất sáu lượt train chính do người dùng chạy; báo cáo kết quả ở liên kết đầu tài liệu. Test EMNLP đã được đánh giá bởi v3, vì vậy đánh giá về sau trên
cùng split phải công bố đây là tập test đã từng xem, không gọi là held-out mới.
