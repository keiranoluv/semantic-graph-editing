# EMNLP Task B v4: cấu hình và launcher W&B đã kiểm thử

**Cập nhật:** người dùng đã hoàn tất 6/6 lượt train chính. Xem
[tổng hợp kết quả validation v4](../emnlp2020-task-b-wandb-v4-results-v1/README.md).
Đã có thêm [kết quả test GPU v4](../emnlp2020-task-b-wandb-v4-test-v1/README.md).
Nội dung dưới đây lưu bằng chứng chuẩn bị launcher và smoke trước lần chạy chính;
assistant chỉ thực hiện kiểm thử CPU/GPU trong giai đoạn chuẩn bị đó.

[Hướng dẫn và lệnh chạy](../../../docs/LEVEL2_EMNLP2020_TASK_B_WANDB_V4.md),
[config](../../../configs/level2/emnlp2020-task-b-wandb-v4.json),
[bằng chứng CPU/GPU](validation.json),
[replay history validation](recipe-decision.json).

V3 dùng batch 128, 12 epoch cố định, AdamW LR 0,001, weight decay 0,01,
hidden 96, GNN 2 layers, seeds 42/43/44, không early stopping/W&B.
V4 giữ các argument khác, thêm **max epochs 20, minimum 6, patience 3,
min_delta 0,001**, monitor val token loss mỗi epoch. Checkpoint vẫn lấy loss
thấp nhất thực tế; min_delta chỉ điều khiển patience.

Replay sáu history validation v3 dừng ở epoch 9–11 và giữ nguyên best epochs
MLP 6/8/8, GNN 8/7/6. Đây là phân tích loss đã có, không phải chạy lại train
hoặc bằng chứng rằng accuracy v4 tăng. Không chọn arguments bằng test metrics.
Test đã từng được đánh giá trong v3; v4 không được mô tả là test mới chưa xem.

CPU contracts: **60 pass, 1 CUDA skip**, gồm 7 test v4. Kiểm fake W&B online-only,
stable IDs, epoch/per-type metric logging, fail trước gradient khi W&B sai mode,
mất kết nối và resume đúng optimizer/RNG/best/early-stop, chống sửa artifact,
resume qua commit chỉ đổi báo cáo. Không tạo W&B run thật trong unit tests.

CPU/GPU smoke: MLP và GNN mỗi model 1 epoch, seed 42, 8 train +4 diagnostic
source-disjoint lấy từ official train, không test. Batch argument 128 nhưng
batch thực tế 8/4; không dùng smoke để khẳng định VRAM batch 128 hoặc accuracy.
Batch 128 giữ từ v3 đã chạy, không thay đổi cùng lúc với stopping policy.
W&B không khởi tạo trong cả hai smoke; kết nối/auth W&B thật được kiểm khi
người dùng chạy main, trước gradient đầu tiên. GPU smoke chạy sau commit
`2821bbd`; output smoke riêng, không đưa vào main metrics.

Main online W&B project `semantic-graph-editing`, một run/model/seed, group theo
experiment ID. Log loss, LR, six-type P/R/F1, micro/macro, exact/valid rate,
OOV/errors, stopping state và GPU memory mỗi epoch; summary của best checkpoint.
Không upload dataset/graphs/instructions/vocabulary/checkpoints/code lên W&B.

Queue chính tự chạy **MLP/42 → MLP/43 → MLP/44 → GNN/42 → GNN/43 → GNN/44**,
một model tại một thời điểm, không tự test. Output:
`runs/level2/emnlp2020-task-b-wandb-v4/`. Resume theo `last.pt` ở epoch boundary,
W&B ID giữ nguyên; lượt hoàn tất được kiểm hash rồi bỏ qua. Khóa checkpoint
chỉ tạo sau đủ sáu lượt. Source/config/manifest/results v2/v3, MagicBrush,
`src/` và notebook cá nhân được giữ nguyên.
